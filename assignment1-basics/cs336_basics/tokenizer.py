
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None 
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        #vocab is a dict mapping id->bytes_combination
        #we also need a dict to map bytes_combination->id
        self.byte_2_id_vocab = {v:k for k,v in vocab.items()}
        #we want to know the ranking of merge rules
        self.merges_ranking = {merge:idx for idx, merge in enumerate(merges)}
        #finanlly we need to handle special tokens
        if special_tokens:
            self.special_tokens_2_bytes = [token.encode('UTF-8') for token in special_tokens]
        else:
            self.special_tokens_2_bytes = []
        #add the special tokens to vocab
        for special_token in self.special_tokens_2_bytes:
            if special_token not in self.byte_2_id_vocab:
                n = len(self.vocab)
                self.vocab[n] = special_token
                self.byte_2_id_vocab[special_token] = n
        

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method that loads a tokenizer from serialized files.

        Args:
            vocab_filepath: Path to a pickle file containing a dict[int, bytes] (or bytes-like).
            merges_filepath: Path to a pickle file containing a list[tuple[bytes, bytes]] (or str-like).
            special_tokens: Optional list[str] to be registered/appended to the vocabulary.

        Returns:
            An initialized Tokenizer instance.
        """
        import pickle

        # Load and normalize vocab: keys -> int, values -> bytes
        with open(vocab_filepath, "rb") as vf:
            raw_vocab = pickle.load(vf)

        norm_vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            kid = int(k)
            if isinstance(v, str):
                v = v.encode("utf-8")
            norm_vocab[kid] = v

        # Load and normalize merges: ensure tuples of bytes
        with open(merges_filepath, "rb") as mf:
            raw_merges = pickle.load(mf)

        norm_merges: list[tuple[bytes, bytes]] = []
        for a, b in raw_merges:
            if isinstance(a, str):
                a = a.encode("utf-8")
            if isinstance(b, str):
                b = b.encode("utf-8")
            norm_merges.append((a, b))

        return cls(norm_vocab, norm_merges, special_tokens)
        
    def encode(self, text: str) -> list[int]:
        res_token_ids = []
        pretokenizeiton = self.pre_tokenization(text, self.special_tokens)
        for part in pretokenizeiton:
            if self.special_tokens and part in self.special_tokens:
                special_id = self.byte_2_id_vocab[part.encode('UTF-8')]
                res_token_ids.append(special_id)
            else:
                res_token_ids.extend(self.encode_text(part))
        return res_token_ids
                                                         

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    def decode(self, ids: list[int]) -> str:
        #transform to byte_list
        byte_list = b''.join(self.vocab[id_] for id_ in ids)
        return byte_list.decode('UTF-8', errors='replace')

    def encode_text(self, pre_token: str):
        '''
        encode a single pre-token (normal text, not special tokens) to token ids
        '''
        def word_2_byte(word: str) -> tuple[bytes, ...]:
            word_decoded = list(word.encode('UTF-8'))
            #split the bytes
            word_byte = [bytes([b]) for b in word_decoded]
            return tuple(word_byte)
        
        word_byte = word_2_byte(pre_token)
        word_byte_after_merge = self.apply_merge(word_byte)
        token_ids = []
        for merged_bytes in word_byte_after_merge:
            id_ = self.byte_2_id_vocab[merged_bytes]
            token_ids.append(id_)
        return token_ids

    def apply_merge(self, word_byte):
        word = list(word_byte)
        def get_pairs(word):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        word_pairs = get_pairs(word)
        if not word_pairs:
                return word
        while True:
            #find the minimum number of pair's ranking
            bigram = min(word_pairs, key=lambda pair: self.merges_ranking.get(pair, float('inf')))
            if bigram not in self.merges_ranking:
                break
            idx = 0
            new_byte_token = []
            first, second = bigram
            while idx < len(word):
                try:
                    #find the nearest 'first' byte
                    first_nearest = word.index(first, idx)
                except ValueError:
                    new_byte_token.extend(word[idx:])
                    break
                else:
                    new_byte_token.extend(word[idx:first_nearest])
                    idx = first_nearest
                if word[first_nearest] == first and first_nearest+1 < len(word) and word[first_nearest+1] == second:
                    new_byte_token.append(first+second)
                    idx += 2
                else:
                    new_byte_token.append(word[first_nearest])
                    idx += 1
            word = new_byte_token
            if len(word) == 1:
                break
            else:
                word_pairs = get_pairs(word)
        return word

            


    def pre_tokenization(self, s: str, special_token: list[str]) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # ① 没有 special 也要按正则切
        if not special_token:
            return re.findall(PAT, s)

        # ③ 长→短排序，防止短的抢先匹配
        toks = sorted(special_token, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in toks)
        parts = re.split(f"({union})", s)

        out = []
        st = set(special_token)
        for part in parts:
            if not part:
                continue
            # ② special 只作为边界，完全跳过
            if part in st:
                out.append(part)
            else:
                out.extend(re.findall(PAT, part))
        return out