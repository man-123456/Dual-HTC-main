# from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .graph import GraphEncoder
from .loss import multilabel_categorical_crossentropy
from utils import get_hierarchy_info
import os
import pickle


class BertPoolingLayer(nn.Module):
    def __init__(self, config, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100
        self.FC = nn.Linear(300, 768)
        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, 768)) for K in self.filter_sizes]
        )

        self.gru = nn.GRU(input_size=768, hidden_size=384, num_layers=2,
                          batch_first=True, bidirectional=True)

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]

        else:
            x = x.mean(dim=1)
        return x


class BertOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NTXent(nn.Module):

    def __init__(self, config, tau=1.):
        super(NTXent, self).__init__()
        self.tau = tau
        self.norm = 1.
        self.transform = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, labels=None):
        x = self.transform(x)
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss


class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    past_key_values = None
    attentions = None
    cross_attentions = None
    input_embeds = None


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            embedding_weight=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if embedding_weight is not None:
            if len(embedding_weight.size()) == 2:
                embedding_weight = embedding_weight.unsqueeze(-1)
            inputs_embeds = inputs_embeds * embedding_weight
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_weight=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            embedding_weight=embedding_weight,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, inputs_embeds) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            inputs_embeds=inputs_embeds,
        )


class LabelAware(nn.Module):  # Compute the label aware embedding
    def __init__(self, label_embedding_size, attn_hidden_size, embed_dim, head, qdim=None, kdim=None, dropout=0.3):
        super(LabelAware, self).__init__()
        self.label_embedding_size = label_embedding_size
        self.attn_hidden_size = attn_hidden_size
        # self.multi_attn_block = MultiAttBlock(self.label_embedding_size, head, self.label_embedding_size, self.label_embedding_size)

        #
        self.num_heads = head
        self.head_dim = embed_dim // head
        self.embed_dim = embed_dim

        self.q_embed_size = qdim if qdim else embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = kdim if kdim else embed_dim
        self.dropout = dropout
        self.head_scale = self.head_dim ** -0.5

        # modify the size here
        d_hq = embed_dim // head
        d_hv = embed_dim // head

        self.query_heads = nn.Linear(self.q_embed_size, self.q_embed_size, bias=True)
        self.key_heads = nn.Linear(self.k_embed_size, self.k_embed_size, bias=True)
        self.value_heads = nn.Linear(self.v_embed_size, self.v_embed_size, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.query_block = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.query_heads.reset_parameters()
        self.key_heads.reset_parameters()
        self.value_heads.reset_parameters()

        self.out_proj.reset_parameters()
        self.query_block.reset_parameters()

    def multiAttn(self, Q, K, V, key_padding_mask=None):
        # Q: batch_size * num_labels * embed_dim
        # K: batch_size * seq_len * embed_dim
        # V: batch_size * seq_len * embed_dim
        # key_padding_mask: batch_size * seq_len

        Q_proj = self.query_heads(Q)  # batch_size * num_labels * embed_dim
        K_proj = self.key_heads(K)  # batch_size * seq_len * embed_dim
        V_proj = self.value_heads(V)  # batch_size * seq_len * embed_dim

        bsz, num_labels, embed_dim = Q_proj.shape
        bsz, seq_len, embed_dim = K_proj.shape

        Q_proj = Q_proj.transpose(0, 1).reshape(num_labels, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                                           1)  # (bsz * num_heads) * num_labels * d_hq
        K_proj = K_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                                        1)  # (bsz * num_heads) * seq_len * d_hq
        V_proj = V_proj.transpose(0, 1).reshape(seq_len, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                                        1)  # (bsz * num_heads) * seq_len * d_hv

        scores = torch.bmm(Q_proj,
                           K_proj.transpose(-2, -1)) * self.head_scale  # (bsz * num_heads) * num_labels * seq_len

        # check if the scores between different batches are the same
        # print(np.isclose(scores[0].cpu().detach().numpy(), scores[self.num_heads].cpu().detach().numpy()).all())

        if key_padding_mask is not None:
            # Reshape the key_padding_mask to have shape (bsz * num_heads, 1, seq_len) to enable broadcasting
            scores = scores.view(bsz, self.num_heads, num_labels, seq_len)
            # filp the mask
            key_padding_mask = key_padding_mask.eq(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # bsz * 1 * 1 * seq_len
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
            scores = scores.view(bsz * self.num_heads, num_labels, seq_len)

        attn_weights = F.softmax(scores, dim=-1)  # (bsz * num_heads) * num_labels * seq_len
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, V_proj)  # (bsz * num_heads) * num_labels * d_hv
        attn = attn.transpose(0, 1).reshape(num_labels, bsz, self.embed_dim).transpose(0,
                                                                                       1)  # bsz * num_labels * embed_dim

        # check if the attention between different batches are the same
        # print(np.isclose(attn[0].cpu().detach().numpy(), attn[1].cpu().detach().numpy()).all())
        attn = self.out_proj(attn)  # bsz * num_labels * embed_dim

        attn_weights = attn_weights.view(bsz, self.num_heads, num_labels, seq_len)

        return attn, attn_weights

    def forward(self, input_data, label_repr, input_data_mask=None, label_repr_mask=None):
        # input_data: batch_size * seq_len * hidden_size
        # label_repr: batch_size * num_labels * label_embedding_size
        # input_data_mask: batch_size * seq_len

        if input_data_mask is not None:
            input_data = input_data * input_data_mask.unsqueeze(-1)
        if label_repr_mask is not None:
            label_repr = label_repr * label_repr_mask.unsqueeze(-1)

        label_aware, attns = self.multiAttn(label_repr, input_data, input_data, input_data_mask)

        return label_aware, attns


class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, tokenizer, cls_loss=True, contrast_loss=True, graph=False, layer=1, data_path=None,
                 multi_label=False, lamb=1, lamb2=1, threshold=0.01, tau=1, label_cpt=None):
        super(ContrastModel, self).__init__(config)
        self.num_labels = config.num_labels
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='./PretrainedModel/bert-base-uncased',
        #                                          add_prefix_space=True)
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.contrastive_lossfct = NTXent(config)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.token_classifier = BertOutputLayer(config)

        self.graph_encoder = GraphEncoder(config, self.tokenizer, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau)
        self.lamb = lamb
        self.lamb2 = lamb2

        self.init_weights()
        self.multi_label = multi_label

        self.label_aware = LabelAware(config.hidden_size, config.hidden_size, config.hidden_size, head=4)  # 1
        self.contrast_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True) #1

        self.classifier1 = nn.Linear(config.hidden_size * config.num_labels, config.hidden_size) #1
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels) #1

        # label_depth
        hiera, _label_dict, r_hiera, label_depth = get_hierarchy_info(label_cpt)
        with open(os.path.join(data_path, 'new_label_dict.pkl'), 'rb') as f:
            label_dict = pickle.load(f)
        label_depth = [label_depth[k] for k, v in label_dict.items()]
        label_depth = np.array(label_depth, dtype=np.int32)

        self.label_depth = label_depth

    def hamming_distance_by_matrix_weighted_by_depth_1(self, labels, label_depth):
        # labels: batch_size, num_labels
        # label_depth: num_labels -> depth
        # make it more efficient and times the label_depth[i]
        depths = torch.tensor(label_depth, dtype=torch.float32, device=self.device)
        depths = torch.max(depths) - depths + 1
        # depths = torch.exp(1 / (torch.max(depths) - depths + 1))
        return torch.matmul(labels * depths, (1 - labels).T) + torch.matmul((1 - labels) * depths, labels.T), torch.sum(depths)

    def label_contrastive_loss(self, label_embeddings, gold_labels, batch_idx, hamming_dist, depth_sum):
        # label_embeddings: (Positive samples) * hidden_size
        # gold_labels: Positive, ranged from 0 to num_labels
        # batch_idx: Positive samples, record the batch idx of each label embeddings
        # hamming_dist: batch_size * batch_size, hamming distance between batches

        # create a expanded hamming distance matrix with the same size with batch_idx
        expanded_hamming_dist = hamming_dist[batch_idx, :][:, batch_idx]

        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        def exp_cosine_sim(x1, x2, eps=1e-15, temperature=1.0):
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = x2.norm(p=2, dim=1, keepdim=True)
            return torch.exp(
                torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
            )

        def exp_sim(x1, x2, eps=1e-15, temperature=1):
            return torch.exp(
                torch.matmul(x1, x2.t()) / temperature
            )

        def cosine_sim(x1, x2, eps=1e-15, temperature=1):
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = x2.norm(p=2, dim=1, keepdim=True)
            return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

        for i in range(self.num_labels):
            # get the positive samples
            pos_idx = (gold_labels == i).nonzero().squeeze(1)
            if pos_idx.numel() == 0:
                continue
            # get the negative samples
            neg_idx = (gold_labels != i).nonzero().squeeze(1)
            if neg_idx.numel() == 0:
                continue

            pos_samples = label_embeddings[pos_idx]  # shape: (num_pos, hidden_size)
            neg_samples = label_embeddings[neg_idx]  # shape: (num_neg, hidden_size)
            size = neg_samples.size(0) + 1

            pos_weight = 1 - expanded_hamming_dist[pos_idx, :][:, pos_idx] / depth_sum  # shape: (num_pos, num_pos)
            neg_weight = expanded_hamming_dist[pos_idx, :][:, neg_idx]  # shape: (num_pos, num_neg)
            pos_dis = exp_cosine_sim(pos_samples, pos_samples) * pos_weight
            neg_dis = exp_cosine_sim(pos_samples, neg_samples) * neg_weight

            denominator = neg_dis.sum(1) + pos_dis
            loss += torch.mean(torch.log(denominator / (pos_dis * size)))
        loss = loss / self.num_labels

        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,

        )

        hidden_last, pooled, hidden_all = outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states
        hidden_cls, encode_out = hidden_last[:, 0, :], hidden_last[:, 1:, :]
        pooled_output = self.dropout(hidden_cls)

        loss = 0
        contrastive_loss = None
        contrast_logits = None

        logits = self.classifier(pooled_output)

        if self.training:
            contrast_mask, graph_inputs = self.graph_encoder(outputs['inputs_embeds'],
                                                             attention_mask, labels,
                                                             lambda x: self.bert.embeddings(x)[0])

            # begin
            sent_inputs_mask = attention_mask[:, 1:]
            label_aware_embedding, attns = self.label_aware(encode_out, graph_inputs, sent_inputs_mask)
            proj_label_embedding = self.dropout(label_aware_embedding)

            fusion_label_embedding = torch.cat([proj_label_embedding, graph_inputs],
                                               dim=-1)  # batch_size * num_labels * (hidden_size * 2)
            fusion_attn_weights = self.contrast_proj(
                fusion_label_embedding)  # batch_size * num_labels * hidden_size
            fusion_attn_weights = torch.softmax(fusion_attn_weights,
                                                dim=-1)  # batch_size * num_labels * hidden_size
            fusion_attn_weights = torch.bmm(fusion_attn_weights,
                                            encode_out.transpose(1, 2))  # batch_size * num_labels * seq_len

            label_specifc_embedding = torch.bmm(fusion_attn_weights,
                                                encode_out)  # batch_size * num_labels * bert_hidden_size

            features = label_specifc_embedding
            label_aware_embedding = self.dropout(features)

            cls_embedding = label_aware_embedding.view(-1,
                                                       label_aware_embedding.shape[1] *
                                                       label_aware_embedding.shape[
                                                           2])
            # label_aware_embedding = self.dropout(label_aware_embedding)
            ## TODO try to add dropout here
            intermediate_embedding = self.classifier1(cls_embedding)
            intermediate_embedding = torch.relu(intermediate_embedding)
            logits2 = self.classifier2(intermediate_embedding)

            if labels is not None:
                target = labels.to(torch.float32)
                loss += multilabel_categorical_crossentropy(target, logits.view(-1, self.num_labels))
                loss += multilabel_categorical_crossentropy(target, logits2.view(-1, self.num_labels))

                if True:
                    labels = labels.to(torch.float32)  # batch_size * num_labels

                    hamming_dist, depth_sum = self.hamming_distance_by_matrix_weighted_by_depth_1(labels,
                                                                                                  self.label_depth)  # distance between labels, batch_size * batch_size

                    hamming_dist = hamming_dist.to(self.device)
                    depth_sum = depth_sum.to(self.device)

                    # create indices tensor that repeat batch_size * 1, batch_size * 2, ..., batch_size * num_labels
                    batch_idx = torch.arange(label_aware_embedding.shape[0]).to(self.device)  # shape = (batch_size)
                    batch_idx = batch_idx.unsqueeze(1).expand(-1,
                                                              self.num_labels).flatten()  # shape = (batch_size * num_labels)

                    # flatten the label_aware_embedding into (batch_size * num_labels) * hidden_size
                    # label_aware_embedding_2 = self.fc2(self.gelu(self.fc1(label_aware_embedding))) # (batch_size * num_labels) * hidden_size
                    label_aware_embedding_2 = label_aware_embedding.view(-1, label_aware_embedding.shape[-1])

                    # get the label_aware_embedding for the labels which eq 1
                    mask = labels.to(torch.bool)  # batch_size, num_labels
                    mask = mask.flatten()  # (batch_size * num_labels)

                    # get the indices of the mask which eq 1
                    gold_label = torch.nonzero(mask).squeeze()  # (batch_size * num_labels)
                    # take moduluo of the self.num_labels to get the indices of the labels
                    gold_label = gold_label % self.num_labels  # (batch_size * num_labels)

                    masked_batch_idx = torch.masked_select(batch_idx, mask)  # (batch_size * num_labels)
                    masked_label_aware_embedding = torch.masked_select(label_aware_embedding_2,
                                                                       mask.unsqueeze(-1).expand_as(
                                                                           label_aware_embedding_2)).view(-1,
                                                                                                          label_aware_embedding_2.shape[
                                                                                                              -1])

                    weighted_label_contrastive_loss = self.label_contrastive_loss(masked_label_aware_embedding,
                                                                                  gold_label,
                                                                                  masked_batch_idx, hamming_dist,
                                                                                  depth_sum).to(self.device)
                    loss += weighted_label_contrastive_loss * self.lamb2
                    # end

            contrast_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_weight=contrast_mask,
            )
            contrast_sequence_output = self.dropout(self.pooler(contrast_output[0]))
            contrast_logits = self.classifier(contrast_sequence_output)
            contrastive_loss = self.contrastive_lossfct(torch.cat([pooled_output, contrast_sequence_output], dim=0),)
            #
            loss += multilabel_categorical_crossentropy(target, contrast_logits.view(-1, self.num_labels)) #1

        if contrastive_loss is not None and self.contrast_loss:
            loss += contrastive_loss * self.lamb


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }
