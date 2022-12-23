import os
import sys

import torch
from tqdm import tqdm
from fairseq.data import Dictionary
import codecs
import nltk
from brl.envs.wmt17zh2en_fairseq.subword_nmt.apply_bpe import BPE, read_vocabulary
import jieba


class Tokenizer:
    """a wrapper for yttm (youtokentome) bpe tokenizer."""

    def __init__(self, lang, bpe_fname):
        self.lang = lang
        self.dictionary = self._load_dictionary((os.path.join(bpe_fname, "dict.%s.txt" % lang)))
        self.bpe = BPE(codecs.open(os.path.join(bpe_fname, "code"), encoding='utf-8'), vocab=self.dictionary.symbols)
        print("vocab size: %s" % len(self.dictionary.symbols))

        self.pad = '<PAD>'  # <pad>
        self.pad_id = self.dictionary.bos_index  # 0
        self.unk = '<UNK>'  # <unk>
        self.unk_id = self.dictionary.pad_index  # 1
        self.bos = '<BOS>'
        self.bos_id = self.dictionary.eos_index  # 2
        self.eos = '<EOS>'
        self.eos_id = self.dictionary.unk_index  # 3

    @staticmethod
    def _load_dictionary(filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        return dictionary

    def str2ids(self, s, add_if_not_exist=False, append_eos=True, append_bos=True):
        """s: list of strings"""
        res = []
        for single in s:
            tokens = jieba.lcut(single) if self.lang == "zh" else nltk.word_tokenize(single)
            single = " ".join(tokens)
            single = self.bpe.process_line(single)
            line = self.dictionary.encode_line(single, add_if_not_exist=add_if_not_exist, append_eos=append_eos).tolist()
            line = [self.bos_id] + line if append_bos else line
            res.append(line)
        return res

    def ids2str(self, ids, remove_bpe=True, lower_case=True):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if isinstance(ids, list):
            ids = torch.LongTensor(ids)
        value = self.dictionary.string(ids)
        value = value.lower() if lower_case else value
        if remove_bpe:
            value = value.replace("@@ ", "")
        return [value]

    def __len__(self):
        return len(self.dictionary)

    def vocab(self):
        return self.dictionary.symbols


if __name__ == '__main__':
    lang = sys.argv[1]
    dictionary_dir = sys.argv[2]
    input_file = sys.argv[3]
    tokenizer = Tokenizer(lang, dictionary_dir)
    with open("%s.id" % os.path.basename(input_file), "w", encoding="utf-8") as writer:
        with open(input_file, 'r', encoding="utf-8") as reader:
            print("%s" % (input_file))
            for line in tqdm(reader, desc="str2ids"):
                ids = tokenizer.str2ids([line], append_eos=False, append_bos=False)
                for id_str in ids:
                    writer.writelines("%s\n" % (" ".join(map(str, id_str))))
