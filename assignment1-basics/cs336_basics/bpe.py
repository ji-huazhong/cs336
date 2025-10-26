import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_REGEX = re.compile(PAT)


def pretokenize(text_segments: list[str]) -> list[str]:
    result = []
    for text in text_segments:
        pretokens = re.findall(PAT, text)
        result.extend(pretokens)
    return result


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
    word_sequences = [pretoken.encode("utf-8") for pretoken in pretokens]

    word_count: dict[bytes, int] = defaultdict(int)
    word_bytes: dict[bytes, list[bytes]] = {}
    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word in word_sequences:
        word_count[word] += 1
        bytes_list = [bytes([b]) for b in word]
        word_bytes[word] = bytes_list
        for i in range(len(bytes_list) - 1):
            pair_count[(bytes_list[i], bytes_list[i + 1])] += 1

    merges: list[tuple[bytes, bytes]] = []
    current_vacab_size = init_vocab_size + len(special_tokens)
    while len(merges) < num_merges:
        if not pair_count:
            break

        best_freq = max(pair_count.values())
        best_pairs = [pair for pair, freq in pair_count.items() if freq == best_freq]
        best_pair = max(best_pairs)

        merges.append(best_pair)
        del pair_count[best_pair]

        a, b = best_pair
        new_token = a + b
        vocab[current_vacab_size] = new_token
        current_vacab_size += 1

        affected_words = []
        for word in word_count:
            bytes_list = word_bytes[word]
            for i in range(len(bytes_list) - 1):
                if bytes_list[i] == a and bytes_list[i + 1] == b:
                    affected_words.append(word)

        for word in affected_words:
            bytes_list = word_bytes[word]
            count = word_count[word]

            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i + 1])
                if pair in pair_count:
                    pair_count[pair] -= count
                    if pair_count[pair] <= 0:
                        del pair_count[pair]
            
            new_bytes_list = []
            i = 0
            while i < len(bytes_list):
                if i < len(bytes_list) - 1 and bytes_list[i] == a and bytes_list[i + 1] == b:
                    new_bytes_list.append(new_token)
                    i += 2
                else:
                    new_bytes_list.append(bytes_list[i])
                    i += 1
            word_bytes[word] = new_bytes_list

            for i in range(len(new_bytes_list) - 1):
                pair = (new_bytes_list[i], new_bytes_list[i + 1])
                pair_count[pair] += count

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
        
        # Check that the special token is not in the vocab
        vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
        for word_bytes in vocabs_without_specials:
            assert b"<|" not in word_bytes
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
