from collections.abc import Iterable


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        基于给定词汇表、合并规则列表和特殊令牌，初始化BPE分词器
        
        参数:
            vocab: 映射token ID到字节序列的字典（如{9: b'the'}）
            merges: BPE合并规则列表，每个元素为(子词1字节序列, 子词2字节序列)
            special_tokens: 可选特殊令牌列表（如['<|endoftext|>']），不存在于vocab时会自动添加
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.byte_to_id: dict[bytes, int] = {token: id for id, token in vocab.items()}

        self.special_tokens = special_tokens if special_tokens is not None else []
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        if not self.special_tokens:
            return

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_id:
                new_token_id = len(self.vocab)
                self.vocab[new_token_id] = token_bytes
                self.byte_to_id[token_bytes] = new_token_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """
        从序列化文件加载词汇表和合并规则，构建并返回BPETokenizer实例
        
        参数:
            vocab_filepath: 词汇表文件路径（JSON格式，键为int型ID，值为base64编码的字节序列）
            merges_filepath: 合并规则文件路径（文本格式，每行一个合并对，子词用空格分隔，已按UTF-8编码）
            special_tokens: 可选特殊令牌列表，加载后会添加到词汇表
        
        返回:
            初始化完成的BPETokenizer实例
        """
        import json
        
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                sub1_str, sub2_str = line.split(" ")
                sub1_bytes = sub1_str.encode("utf-8")
                sub2_bytes = sub2_str.encode("utf-8")
                merges.append((sub1_bytes, sub2_bytes))
        return cls(vocab, merges, special_tokens)

    def _split_by_special_token(self, text: str) -> list[str]:
        import re

        if not self.special_tokens:
            segments = [text]
        else:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            segments = re.split(f"({pattern})", text)
            segments = [seg for seg in segments if seg]
        return segments

    def _pretokenize(self, text: str) -> list[bytes]:
        import regex as re
    
        pre_tokens = re.findall(PAT, text)
        return [token.encode("utf-8") for token in pre_tokens]
    
    def _apply_merges(self, pretoken: bytes) -> list[bytes]:
        tokens = [bytes([token]) for token in pretoken]

        for a, b in self.merges: # 按照merges记录的顺序merge token
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == a and
                    tokens[i + 1] == b):
                    merged_token = a + b
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1 
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        segments = self._split_by_special_token(text)
        token_ids = []
        for segment in segments:
            if segment in self.special_tokens:
                token_bytes = segment.encode("utf-8")
                if token_bytes in self.byte_to_id:
                    token_ids.append(self.byte_to_id[token_bytes])
                else:
                    raise ValueError(f"Specail token {segment} not found in vocabulary")
            else:
                pretokens = self._pretokenize(segment)
                for pretoken in pretokens:
                    merged_tokens = self._apply_merges(pretoken)
                    for token_bytes in merged_tokens:
                        if token_bytes in self.byte_to_id:
                            token_ids.append(self.byte_to_id[token_bytes])
                        else:
                            raise ValueError(f"Token {token_bytes} not found in vocabulary")

        return token_ids 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_sequences = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
            else:
                continue
        if byte_sequences:
            merged_bytes = b"".join(byte_sequences)
        else:
            merged_bytes = b""

        try:
            return merged_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return merged_bytes.decode("utf-8", errors="replace")
