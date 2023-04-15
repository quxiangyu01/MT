import torch
from torch import nn
import copy
import numpy as np


def triu_mask(y_mask, device):
    """x: (xlen, ...)"""
    q_mask, k_mask = y_mask.T.unsqueeze(-1), y_mask.T.unsqueeze(1)
    # mask zero-padding
    zero_attn_mask = k_mask.data.eq(0)  # bx1xsk
    zero_attn_mask = zero_attn_mask.expand(q_mask.size(0), q_mask.size(1), k_mask.size(2))  # bxsqxsk

    #  triu_attn_mask
    triu_attn_mask = torch.triu(torch.ones(zero_attn_mask.size(), device=device), 1).bool()

    sent_attn_mat = torch.bmm(q_mask, k_mask)
    mask_label = np.unique(q_mask.cpu().numpy())
    mask_label = np.setdiff1d(mask_label, np.array([0]))
    label_times = np.multiply(mask_label.reshape(1, mask_label.size), mask_label.reshape(mask_label.size, 1))
    label_times = np.setdiff1d(label_times.flatten(), np.diag(label_times, 0))

    # mask sent-padding
    sent_attn_mask_total = torch.zeros(sent_attn_mat.size(), dtype=torch.int64, device=device).bool()
    for label_time in label_times:
        sent_attn_mask_eq = sent_attn_mat.eq(int(label_time))
        sent_attn_mask_total += sent_attn_mask_eq

    mask = zero_attn_mask + sent_attn_mask_total + triu_attn_mask

    return mask  # (xlen, xlen)


def kv_mask(q_mask, k_mask, device):
    q_mask, k_mask = q_mask.T.unsqueeze(-1), k_mask.T.unsqueeze(1)
    # mask zero-padding
    zero_attn_mask = k_mask.data.eq(0)  # bx1xsk
    zero_attn_mask = zero_attn_mask.expand(q_mask.size(0), q_mask.size(1), k_mask.size(2))  # bxsqxsk

    mask_label = np.unique(q_mask.cpu().numpy())
    mask_label = np.setdiff1d(mask_label, np.array([0]))
    label_times = np.multiply(mask_label.reshape(1, mask_label.size), mask_label.reshape(mask_label.size, 1))
    label_times = np.setdiff1d(label_times.flatten(), np.diag(label_times, 0))

    # mask sent-padding
    sent_attn_mat = torch.bmm(q_mask, k_mask)  # bxsqxsk
    sent_attn_mask_total = torch.zeros(sent_attn_mat.size(), dtype=torch.int64, device=device).bool()
    for label_time in label_times:
        sent_attn_mask_eq = sent_attn_mat.eq(int(label_time))
        sent_attn_mask_total += sent_attn_mask_eq

    mask = zero_attn_mask + sent_attn_mask_total
    return mask


def attn(q, k, v, num_heads, mask=None):
    """
    q: (ql, b, dim)
    k: (kl, b, dim)
    v: (vl, b, dim)
    mask: (b, ql, kl)
    """
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    score = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # (b, ql, kl)
    if mask is not None:
        mask = torch.cat([mask] * num_heads, 0)
        score = score.masked_fill(mask, -float('inf'))
    score = score.softmax(-1)
    out = (score @ v)
    out = out.transpose(0, 1)
    return out, score


class NN(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device


class Embedding(NN):
    def __init__(self, vocab_size, dim, padding_idx, dropout):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.uniform_(self.emb.weight, -0.01, 0.01)
        # self.norm = nn.LayerNorm(dim)
        max_len = 1024
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() *
            (-torch.tensor(10000.0).log() / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, x, pos=0):
        x = self.emb(x) * (self.dim ** 0.5)
        x = x + self.pe[pos:x.size(0) + pos]
        x = self.dropout(x)
        # x = self.norm(x)
        return x


class MultiheadAttention(NN):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.q_in = nn.Linear(dim, dim, bias=False)
        self.k_in = nn.Linear(dim, dim, bias=False)
        self.v_in = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, q, k, v, mask=None):
        q_len = q.size(0)
        k_len = k.size(0)
        v_len = v.size(0)
        bsz = q.size(1)

        q = self.q_in(q)
        k = self.k_in(k)
        v = self.v_in(v)

        q = q.reshape(q_len, bsz * self.num_heads, self.head_dim)
        k = k.reshape(k_len, bsz * self.num_heads, self.head_dim)
        v = v.reshape(v_len, bsz * self.num_heads, self.head_dim)

        out, score = attn(q, k, v, self.num_heads, mask)

        out = out.reshape(q_len, bsz, self.dim)
        out = self.out(out)
        return out


class FFN(NN):
    def __init__(self, dim, inner_dim):
        super().__init__()

        self.l1 = nn.Linear(dim, inner_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class EncoderLayer(NN):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):  # 2
        x = x + self.dropout(self.self_attn(x, x, x, mask))  # 3
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class DecoderLayer(NN):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        self.self_attn = MultiheadAttention(dim, num_heads)
        self.ctx_attn = MultiheadAttention(dim, num_heads)
        self.ffn = FFN(dim, dim * 4)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, y, x, decode_triu_mask, decode_mask):  # 4 y, encode_out, decode_triu_mask, decode_mask
        """y_step: (ylen, b, dim)
        x: (xlen, b, dim)"""
        y = y + self.dropout(self.self_attn(y, y, y, decode_triu_mask))
        y = self.norm1(y)
        y = y + self.dropout(self.ctx_attn(y, x, x, decode_mask))  # 5
        y = self.norm2(y)
        y = y + self.dropout(self.ffn(y))
        y = self.norm3(y)
        return y

    '''
    def forward_step(self, y_step, x, cache):
        """y_step: (1, b, dim)
        x: (xlen, b, dim)
        cache: (ylen, b, dim), in which 0 is self_attn_cache, 
                1 is ctx_attn_cache"""

        cache = y_sofar = torch.cat([cache, y_step])
        y_step = y_step + self.dropout(self.self_attn(y_step, y_sofar,
                                                      y_sofar))
        y_step = self.norm1(y_step)

        y_step = y_step + self.dropout(self.ctx_attn(y_step, x, x))
        y_step = self.norm2(y_step)

        y_step = y_step + self.dropout(self.ffn(y_step))
        y_step = self.norm3(y_step)

        return y_step, cache

    def init_cache(self, b, dim):
        """
        cache: (ylen=0, b, dim), actually the y_prev
        """
        cache = torch.empty(0, b, dim).to(self.device)
        return cache
    '''


class Encoder(NN):
    def __init__(self, layer, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])

    def forward(self, x, mask):  # 6
        for l in self.layers:
            x = l(x, mask)  # 7
        return x


class Decoder(NN):
    def __init__(self, layer, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])

    def forward(self, y, x, decode_triu_mask, decode_mask):  # 8
        for l in self.layers:
            y = l(y, x, decode_triu_mask, decode_mask)  # 9
        return y

    '''
    def forward_step(self, y_step, x, cache):
        """y_step: (1, b, dim)
        x: (xlen, b, dim)
        cache: (layer, ylen, b, dim), in which 0 is self_attn_cache, 
                1 is ctx_attn_cache"""
        new_cache = []
        for i, l in enumerate(self.layers):
            y_step, cache_ = l.forward_step(y_step, x, cache[i])
            new_cache.append(cache_)
        return y_step, torch.stack(new_cache)

    def init_cache(self, b, dim):
        """
        cache: (layer, ylen=0, b, dim), actually y_prev
        """
        num_layers = len(self.layers)
        cache = torch.empty(num_layers, 0, b, dim).to(self.device)
        return cache
    '''


class Transformer(NN):
    def __init__(self,
                 src_token_space_size,
                 trg_token_space_size,
                 dim=512,
                 num_layers=6,
                 num_heads=8,
                 dropout=0,
                 pad_id=0,
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.pad_id = pad_id
        # if pad_id != 0:
        #     print('[error] pad_id = {}, but this Transformer model only supports pad_id = 0'.format(pad_id))
        #     assert False

        self.src_emb = Embedding(src_token_space_size,
                                 dim=dim,
                                 padding_idx=self.pad_id,
                                 dropout=dropout)
        self.trg_emb = Embedding(trg_token_space_size,
                                 dim=dim,
                                 padding_idx=self.pad_id,
                                 dropout=dropout)
        self.trg_emb.emb.weight = self.src_emb.emb.weight
        self.proj = nn.Linear(dim, trg_token_space_size, bias=False)
        self.proj.weight = self.trg_emb.emb.weight


        encoder_layer = EncoderLayer(dim, num_heads, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)
        decoder_layer = DecoderLayer(dim, num_heads, dropout)
        self.decoder = Decoder(decoder_layer, num_layers)

        # hack: name placeholders for loading models from longtu's legacy code; not used at all
        self.src_tok = None  # src_tok
        self.trg_tok = None  # trg_tok
        print("\ncreated a transformer_best\n")

    def forward_batch(self, x, y):
        mask = self.get_patch_seq(x, y)
        if mask is not None:
            encode_mask = kv_mask(mask[0], mask[0], x.device)
            decode_triu_mask = triu_mask(mask[1], x.device)
            decode_mask = kv_mask(mask[1], mask[0], x.device)
        else:
            encode_mask, decode_triu_mask, decode_mask = None, None, None

        x = self.src_emb(x)
        encode_out = self.encoder(x, encode_mask)
        y = self.trg_emb(y)
        decode_out = self.decoder(y, encode_out, decode_triu_mask, decode_mask)
        proj_out = self.proj(decode_out)

        return proj_out

    def get_patch_seq(self, x, y):

        def get_masks(x_list, last=False):
            mask_token = [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 11, -11, 13, -13, 17, -17, 19, -19, 23, -23, 29, -29, 31, -31, 37, -37, 41, -41, 43, -43, 47, -47, 59, -59, 61, -61, 67, -67, 71, -71, 73, -73, 79, -79, 83, -83, 89, -89, 97, -97]
            x_masks = []
            for x in x_list:
                temp_2 = []
                temp_3 = []
                x_mask = len(x) * [0]
                for i, e in enumerate(x):
                    if int(e) == 2:
                        temp_2.append(i)
                    elif int(e) == 3:
                        temp_3.append(i)
                if last and len(temp_2) != len(temp_3):
                    temp_3.append(len(x) - 1)
                for i, t_2 in enumerate(temp_2):
                    t_3 = temp_3[i]
                    x_mask[t_2:t_3 + 1] = [mask_token[i]] * (t_3 - t_2 + 1)
                x_masks.append(x_mask)
            return x_masks

        x_list, y_list = x.T.tolist(), y.T.tolist()
        x_masks = get_masks(x_list)
        y_masks = get_masks(y_list, last=True)

        src_tensor = torch.tensor(x_masks, dtype=torch.float, device=self.device).T
        trg_tensor = torch.tensor(y_masks, dtype=torch.float, device=self.device).T

        return src_tensor, trg_tensor

