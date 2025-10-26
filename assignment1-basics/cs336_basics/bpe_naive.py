import regex as re
from collections import defaultdict

from .profiler import do_cprofile


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_REGEX = re.compile(PAT)


def pretokenize(text_segments: list[str]) -> list[str]:
    result = []
    for text in text_segments:
        pretokens = re.findall(PAT, text)
        result.extend(pretokens)
    return result


def get_best_pair(pair_count):
    best_freq = max(pair_count.values())
    best_pairs = [pair for pair, freq in pair_count.items() if freq == best_freq]
    return max(best_pairs)


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
    
    with open(input_path) as f:
        text = f.read()

    if special_tokens:
        escaped_specials = [re.escape(token) for token in special_tokens]
        pattern = re.compile(f"({'|'.join(escaped_specials)})")
        text_segments = pattern.split(text)
        text_segments = [seg for seg in text_segments if seg not in special_tokens]
    else:
        text_segments = [text]

    pretokens: list[str] = pretokenize(text_segments)

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

        best_pair = get_best_pair(pair_count)

        merges.append(best_pair)

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
                    pair_count[best_pair] -= 1
                    if best_pair == 0:
                        del pair_count[best_pair]
                    
                    if left is not None:
                        old_pair = (left, a)
                        pair_count[old_pair] -= 1
                        pair_count[(left, new_token)] += 1
                        if pair_count[old_pair] == 0:
                            del pair_count[old_pair]

                    if i + 1 < cur_len:
                        right = seq[i + 1]
                        old_pair = (b, right)
                        pair_count[old_pair] -= 1
                        pair_count[(new_token, right)] += 1
                        if pair_count[old_pair] == 0:
                            del pair_count[old_pair]
                else:
                    i += 1
    
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

        temp_file.write("abbc<|endoftext|>abdc")
        temp_file.close()

        vocab, merges = train_bpe(temp_file.name, 280, special_tokens=["<|endoftext|>"])
        print(f"vocab = {vocab}\n merges={merges}", flush=True)

        print(f"vocab = {vocab}\n merges={merges}", flush=True)
        
        # Check that the special token is not in the vocab
        vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
        for word_bytes in vocabs_without_specials:
            assert b"<|" not in word_bytes
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
