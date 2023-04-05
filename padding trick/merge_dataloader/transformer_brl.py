import torch
from torch import nn
import copy


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
        x = self.emb(x) * (self.dim**0.5)
        x = x + self.pe[pos:x.size(0) + pos]
        x = self.dropout(x)
        # x = self.norm(x)
        return x


def attn(q, k, v, mask=None):
    """
    q: (ql, b, dim)
    k: (kl, b, dim)
    v: (vl, b, dim)
    mask: (b, ql, kl)
    """
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    score = (q @ k.transpose(-2, -1)) / (q.size(-1)**0.5)  # (b, ql, kl)
    if mask is not None:
        score = score.masked_fill(mask, -float('inf'))
    score = score.softmax(-1)
    out = (score @ v)
    out = out.transpose(0, 1)
    return out, score


def triu_mask(x):
    """x: (xlen, ...)"""
    xlen = x.size(0)
    mask = torch.triu(torch.ones(xlen, xlen, device=x.device), 1).bool()
    return mask  # (xlen, xlen)


def kv_mask(x, padding_idx, num_heads):
    """x: (xlen, b)"""
    xlen = x.size(0)
    mask = x.eq(padding_idx).T.unsqueeze(1)  # (b, 1, xlen)  1 mask
    mask = torch.stack([mask] * num_heads, 1).view(-1, 1, xlen)
    return mask


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

        out, score = attn(q, k, v, mask)

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

    def forward(self, y, x, mask):  # 4
        """y_step: (ylen, b, dim)
        x: (xlen, b, dim)"""
        y = y + self.dropout(self.self_attn(y, y, y, triu_mask(y)))
        y = self.norm1(y)
        y = y + self.dropout(self.ctx_attn(y, x, x, mask))  # 5
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
        y_step = y_step + self.dropout(self.self_attn(y_step, y_sofar, y_sofar))
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

    def forward(self, y, x, mask):  # 8
        for l in self.layers:
            y = l(y, x, mask)  # 9
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
                 pad_id=0,  # TODO: disable default parameter here -- user needs to explicitly specify the padding id
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.pad_id = pad_id
        # if pad_id != 0:
        #    print('[error] pad_id = {}, but this Transformer model only supports pad_id = 0'.format(pad_id))
        #    assert False

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

        #hack: name placeholders for loading models from longtu's legacy code; not used at all
        self.src_tok = None #src_tok
        self.trg_tok = None #trg_tok
        print("\ncreated a transformer_brl\n")

    def forward_batch(self, x, y):
        # used for training
        mask = kv_mask(x, self.pad_id, self.num_heads)  # 10
        x = self.src_emb(x)
        x = self.encoder(x, mask)  # 11
        y = self.trg_emb(y)
        y = self.decoder(y, x, mask)  # 12
        y = self.proj(y)
        return y

''' 
    @torch.no_grad()
    def forward(self, s, k, lenpen=0.6, score_type='score_current'):
        """s: list of strings
        score_type: one of the score types: score_current, score_sofar, score_y_rm_sofar
        """
        idss = self.src_tok.str2index(s)
        index = Dataset.pad(idss, self.src_tok, device=self.device)
        hyp_index = self.beam_search(index, k, lenpen, score_type)

        idss = Dataset.unpad(hyp_index, self.trg_tok)
        hyp = self.trg_tok.index2str(idss)
        return hyp

    def beam_search(self, x, k, lenpen, score_type, max_len_ratio=2):
        x = self.src_emb(x)
        x = self.encoder(x)

        beams = self.init_beams(x)
        beams = self.beam_step(beams, x, k)
        with tqdm(range(x.size(0) * max_len_ratio), desc='beam search',
                  leave=False) as beam_tqdm:
            for i in beam_tqdm:
                beams = self.beam_step(beams, x, k)
                beams = self.beam_trim(beams, k, lenpen, score_type)
                non_eos_nodes = [
                    n for b in beams for n in b if n.id != self.trg_tok.eos_id
                ]
                beam_tqdm.set_postfix({'alive': len(non_eos_nodes)})
                if len(non_eos_nodes) == 0:
                    break
        hyp_index = self.finalize(beams, lenpen=lenpen, score_type=score_type)#'score_current')
        return hyp_index

    def init_beams(self, x):
        _, b, dim = x.size()
        cache = self.decoder.init_cache(b, dim)  # (n, l, b, dim)
        beams = []

        for bid in range(b):
            node = Node(
                parent=None,
                children=[],
                x=x[:, bid],
                cache=cache[:, :, bid],
                id=torch.tensor([self.trg_tok.bos_id], device=self.device),
                lprob=torch.tensor([0.], device=self.device),
                max_lprob=torch.tensor([0.], device=self.device),
            )
            beam = [node]
            beams.append(beam)
        return beams

    def beam_step(self, beams, x, k):
        # select non-eos nodes
        def not_eos_node(n):
            if n.id == self.trg_tok.eos_id:
                return False
            else:
                return True

        all_nodes = [n for b in beams for n in b]
        non_eos_nodes = list(filter(not_eos_node, all_nodes))

        # gather tensors
        y_step = torch.tensor([[n.id for n in non_eos_nodes]],
                              device=self.device)
        x = torch.stack([n.x for n in non_eos_nodes], dim=1)
        cache = torch.stack([n.cache for n in non_eos_nodes], dim=2)

        # forward_step
        y_step = self.trg_emb(y_step, non_eos_nodes[0].pos)
        y_step, cache = self.decoder.forward_step(y_step, x, cache)
        y_lprob = self.proj(y_step).log_softmax(-1)  # (1, b, vocab_size)

        # make new beams
        new_beams = []
        bid = -1
        for beam in beams:
            new_beam = []
            for node in beam:
                if not_eos_node(node):
                    bid += 1
                    lprob_k, id_k = y_lprob[0, bid].topk(k)
                    for kid in range(k):
                        new_node = Node(
                            parent=node,
                            children=[],
                            x=node.x,
                            cache=cache[:, :, bid],
                            id=id_k[kid],
                            lprob=lprob_k[kid],
                            max_lprob=lprob_k[0],
                        )
                        node.children.append(new_node)
                        new_beam.append(new_node)
                else:
                    new_beam.append(node)
            new_beams.append(new_beam)
        return new_beams

    def beam_trim(self, beams, k, lenpen, score_type):#='score_current'):
        new_beams = []
        for beam in beams:
            if score_type == 'score_lprob':
                key = lambda n: n.score_lprob()
            elif score_type == 'score_current':
                key = lambda n: n.score_current(lenpen)
            elif score_type == 'score_sofar':
                key = lambda n: n.score_sofar(lenpen)
            elif score_type == 'score_y_rm_sofar':
                key = lambda n: n.score_y_rm_sofar(lenpen)
            elif score_type == 'variance_regularizer':
                key = lambda n: n.score_variance_regularizer(lenpen)
            elif score_type == 'local_consistency':
                key = lambda n: n.score_local_consistency(lenpen)
            elif score_type == 'max_regularizer':
                key = lambda n: n.score_max_regularizer(lenpen)
            elif score_type == 'squared_regularizer':
                key = lambda n: n.score_squared_regularizer(lenpen)
            else:
                raise ValueError('not implemented')

            beam = sorted(beam, key=key, reverse=True)
            new_beam = beam[:k]
            new_beams.append(new_beam)

            for n in new_beam:
                if n.id == self.trg_tok.eos_id:
                    n.cache = None

            for n in beam[k:]:
                n.cache = None
        return new_beams

    def finalize(self, beams, lenpen, score_type):
        beams = self.beam_trim(beams, 1, lenpen=lenpen, score_type=score_type)
        nodes = [n for b in beams for n in b]

        def get_idss(node):
            idss = []
            while node.parent is not None:
                idss.append(node.id)
                node = node.parent
            idss = list(reversed(idss))
            return idss

        hyp_idss = list(map(get_idss, nodes))
        hyp_index = Dataset.pad(hyp_idss, self.trg_tok, device=self.device)
        return hyp_index


class Node:
    __slots__ = [
        'parent', 'children', 'x', 'cache', 'id', 'lprob', 'lprob_sofar',
        'pos', 'max_lprob', 'y_rm_sofar'
    ]

    def __init__(self, parent, children, x, cache, id, lprob, max_lprob):
        """
        x: (xlen, dim)
        cache: (layer, ylen, dim)
        id: (1)
        lprob: (1)
        """

        self.parent = parent
        self.children = children
        self.x = x
        self.cache = cache
        self.id = id
        self.lprob = lprob
        self.max_lprob = max_lprob

        if self.parent is not None:
            self.lprob_sofar = self.parent.lprob + self.lprob
            self.pos = self.parent.pos + 1
            self.y_rm_sofar = self.lprob_sofar - self.max_lprob
        else:
            self.lprob_sofar = lprob  # 0
            self.pos = 0

    def __repr__(self) -> str:
        return f'<node id:{self.id} lprob:{self.lprob}>'

    # bojun
    def score_lprob(self):
        return self.lprob

    def score_current(self, lenpen=0.6):  # the one used in Transformer paper
        score = self.lprob # / (self.pos + 1)
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    # the following score functions are not double checked yet
    def score_sofar(self, lenpen=0.6):
        score = self.lprob_sofar / (self.pos + 1)
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    def score_y_rm_sofar(self, lenpen=0.6):
        score = self.y_rm_sofar  / (self.pos + 1)
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    def score_variance_regularizer(self, lenpen=0.6):
        u = self.lprob_sofar / (self.pos + 1)
        R = ((self.lprob - u) / (self.pos + 1))**2
        score = self.y_rm_sofar / (self.pos + 1) - R
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    def score_local_consistency(self, lenpen=0.6):
        R = ((self.lprob - self.parent.lprob_sofar) / (self.pos + 1))**2
        score = self.y_rm_sofar / (self.pos + 1) - R
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    def score_max_regularizer(self, lenpen=0.6):
        R = -float('inf')
        node = self
        while node.parent is not None:
            if node.lprob > R:
                R = node.lprob
            node = node.parent
        score = self.y_rm_sofar / (self.pos + 1) - R
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp

    def score_squared_regularizer(self, lenpen=0.6):
        R = (self.lprob)**2
        score = self.y_rm_sofar / (self.pos + 1) - R
        lp = (((5 + (self.pos + 1))**lenpen) / (6**lenpen))
        return score / lp
'''