import regex as re
from collections import defaultdict

from .profiler import do_cprofile


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


@do_cprofile("./train_bpe.prof")
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    init_vocab_size = 256
    num_merges = vocab_size - init_vocab_size - len(special_tokens)
    if num_merges < 0:
        raise ValueError(f"vocab_size {vocab_size} is too small to hold the initialize tokens and special tokens.")
    vocab = {i: bytes([i]) for i in range(init_vocab_size)}
    for i in range(len(special_tokens)):
        vocab[i+init_vocab_size] = special_tokens[i].encode("utf-8")
    
    # 2. read bpe training data
    with open(input_path) as f:
        raw_input = f.read()
    # 3. pre-tokenization without special tokens
    pretokens: list[str] = pretokenize(raw_input, special_tokens)

    # token_sequences = [list(pretoken.encode("utf-8")) for pretoken in pretokens]
    token_sequences = [[bytes([b]) for b in pretoken.encode("utf-8")] for pretoken in pretokens]

    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for seq in token_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_count[pair] += 1

    merges: list[tuple[bytes, bytes]] = []
    current_vacab_size = init_vocab_size + len(special_tokens)
    while len(merges) < num_merges:
        if not pair_count:
            break

        best_pair = None
        max_freq = -1
        for pair, freq in pair_count.items():
            if freq > max_freq or (freq == max_freq and pair > best_pair):
                max_freq = pair_count[pair]
                best_pair = pair

        a, b = best_pair
        new_token = a + b
        for seq in token_sequences:
            i = 0
            cur_len = len(seq)
            while i < cur_len - 1:
                if seq[i] == a and seq[i + 1] == b:
                    left = seq[i - 1] if i > 0 else None
                    cur_len -= 1
                    seq[i:i + 2] = [new_token]
                    pair_count[(a, b)] -= 1
                    if pair_count[(a, b)] == 0:
                        del pair_count[(a, b)]
                    
                    if left is not None:
                        old_pair = (left, a)
                        pair_count[old_pair] -= 1
                        if pair_count[old_pair] == 0:
                            del pair_count[old_pair]
                        pair_count[(left, new_token)] += 1

                    if i + 1 < cur_len:
                        right = seq[i + 1]
                        old_pair = (b, right)
                        pair_count[old_pair] -= 1
                        if pair_count[old_pair] == 0:
                            del pair_count[old_pair]
                        pair_count[(new_token, right)] += 1
                else:
                    i += 1
    
        merges.append((a, b))
        vocab[current_vacab_size] = new_token
        current_vacab_size += 1
    return vocab, merges


if __name__ == "__main__":
    import os
    import tempfile

    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            encoding="utf-8",
            suffix=".tmp"
        )

        temp_file.write("abbc")
        temp_file.close()

        vocab, merges = train_bpe(temp_file.name, 280, special_tokens=["<|endoftext|>"])
        print(f"vocab = {vocab}\n merges={merges}", flush=True)
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
