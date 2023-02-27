import youtokentome as yttm


class Tokenizer:
    """a wrapper for yttm (youtokentome) bpe tokenizer."""
    def __init__(self, lang, bpe_fname):
        self.lang = lang
        self.bpe_fname = bpe_fname
        self.bpe = yttm.BPE(self.bpe_fname)

        self.bos = '<BOS>'
        self.bos_id = self.bpe.subword_to_id(self.bos)
        self.eos = '<EOS>'
        self.eos_id = self.bpe.subword_to_id(self.eos)
        self.pad = '<PAD>'
        self.pad_id = self.bpe.subword_to_id(self.pad)
        self.unk = '<UNK>'
        self.unk_id = self.bpe.subword_to_id(self.unk)

    def __deepcopy__(self, memodict={}):
        return self

    def str2ids(self, s, bos=True, eos=True):
        """s: list of strings"""
        index = self.bpe.encode(s, bos=bos, eos=eos)
        return index

    # TODO: ids2str() should return a string (when a single id sequence is input), not a list of strings
    # TODO: ids2str() should automatically detect BOS/EOS and ignore them if they exist
    def ids2str(self, ids):
        """index: list of lists of ids"""
        s = self.bpe.decode(ids)
        return s

    def __len__(self):
        return self.bpe.vocab_size()

    def vocab(self):
        return self.bpe.vocab()