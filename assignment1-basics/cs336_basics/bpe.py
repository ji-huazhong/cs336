import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> list[str]:
    def _pure_pretokenize(text: str):
        return re.findall(PAT, text)

    if len(special_tokens) == 0:
        return _pure_pretokenize(text)

    escaped_specials = [re.escape(token) for token in special_tokens]
    split_pat = re.compile(f"({'|'.join(escaped_specials)})")
    text_fragments = split_pat.split(text)
    text_fragments = [frag for frag in text_fragments if frag not in special_tokens]

    result = []
    for frag in text_fragments:
        pretokens = _pure_pretokenize(frag)
        result.extend(pretokens)
    return result


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    init_vocab_size = 256
    num_merges = vocab_size - init_vocab_size - len(special_tokens)
    if num_merges:
        raise ValueError(f"{vocab_size {vocab_size}} is too small to hold the initialize tokens and special tokens.")
    vocab = {i: bytes([i]) for i in range(init_vocab_size)}
    for i in range(len(special_tokens)):
        vocab[i+init_vocab_size] = special_tokens[i].encode("utf-8")
    
    # 2. read bpe training data
    with open(input_path, "r") as f:
        raw_input = f.read()
    # 3. pre-tokenization without special tokens
    pretokens: list[str] = pretokenize(raw_input)

    token_sequences = [list(pretoken.encode("utf-8")) for pretoken in pretokens]

    merges: list[tuple[bytes, bytes]] = []
    current_vacab_size = init_vocab_size + len(special_tokens)
    while len(merges) < num_merges:
        pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_count[pair] += 1
        
        if not pair_count:
            break
        
        sorted_pairs = sorted(pair_count.items(), key= lambda x: (x[1], x[0]))

        a, b = sorted_pairs[-1][0]
        new_token = a + b

        for i in range(len(token_sequences)):
            seq = token_sequences[i]
            new_seq = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    new_seq.append(new_token)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            token_sequences[i] = new_seq
    
        merges.append((a, b))
        vocab[current_vacab_size] = new_token
        current_vacab_size += 1
    return vocab, merges
