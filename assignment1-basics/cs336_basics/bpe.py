import multiprocessing
import os
import regex as re
from collections import Counter

from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_REGEX = re.compile(PAT)


def read_file_in_chunks(file_path: str, chunk_size: int):
    with open(file_path, encoding="utf-8") as f:
        buffer = ""
        while True:
            data = f.read(chunk_size - len(buffer))
            if not data:
                break
            buffer += data

            while len(buffer) >= chunk_size:
                newline_pos = buffer.rfind("\n", 0, chunk_size)
                if newline_pos == -1:
                    break
                chunk = buffer[:newline_pos + 1]
                yield chunk
                buffer = buffer[newline_pos + 1:]
        if buffer:
            yield buffer


def pretokenize(text_segments: list[str]) -> list[str]:
    if not isinstance(text_segments, list):
        raise ValueError("text_segments must be a list of strings, but got {type(text_segments)}")
    result = []
    for text in text_segments:
        pretokens = re.findall(PAT, text)
        pretokens = [pretoken.encode("utf-8") for pretoken in pretokens]
        result.extend(pretokens)
    return result


def pretokenize_in_chunks(chunk: tuple[str, ...]) -> list[str]:
    text, special_tokens = chunk
    word_freqs = Counter()
    pair_freqs = Counter()

    if special_tokens:
        escaped_specials = [re.escape(token) for token in special_tokens]
        pattern = re.compile(f"({'|'.join(escaped_specials)})")
        text_segments = pattern.split(text)
        text_segments = [seg for seg in text_segments if seg not in special_tokens]
    else:
        text_segments = [text]

    words = pretokenize(text_segments)
    for word in words:
        word_freqs[word] += 1
        byte_pairs = [(word[i:i+1], word[i+1:i+2]) for i in range(len(word) - 1)]
        for pair in byte_pairs:
            pair_freqs[pair] += 1

    return word_freqs, pair_freqs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    file_size = os.path.getsize(input_path)
    if file_size < 1024 * 1024:
        chunk_size = file_size + 1
    else:
        chunk_size = 64 * 1024 * 1024

    print("Start BPE Training.")
    print(f"Input file: {input_path} ({file_size / (1024*1024):.2f}MB)")
    print(f"Target vocab size: {vocab_size}")
    print(f"Chunk size: {chunk_size // (1024 * 1024)}MB")
    vocab: dict[int, bytes] = {}
    token_id = 0
    
    if special_tokens is None:
        special_tokens = []
    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1
    
    for val in range(256):
        vocab[token_id] = bytes([val])
        token_id += 1

    print("Reading input file in chunks...")
    chunks = list(read_file_in_chunks(input_path, chunk_size))
    chunks = [(chunk, special_tokens) for chunk in chunks]
    print(f"Read {len(chunks)} chunks.")

    num_workers = min(len(chunks), os.cpu_count()//2)
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(pretokenize_in_chunks, chunks)

    word_freqs = Counter()
    pair_freqs = Counter()
    for chunk_word_freqs, chunk_pair_freqs in results:
        word_freqs.update(chunk_word_freqs)
        pair_freqs.update(chunk_pair_freqs)

    print(f"Parallel pretokenization done. Total unique words: {len(word_freqs)}, total unique byte pairs: {len(pair_freqs)}")

    word_bytes: dict[bytes, list[bytes]] = {}
    for word in word_freqs:
        word_bytes[word] = [bytes([b]) for b in word]

    merges: list[tuple[bytes, bytes]] = []
    total_steps = vocab_size - len(vocab)
    pbar = tqdm(total=total_steps, desc="BPE Merges")
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break

        pbar.update(1)

        best_freq = max(pair_freqs.values())
        best_pairs = [pair for pair, freq in pair_freqs.items() if freq == best_freq]
        best_pair = max(best_pairs)

        merges.append(best_pair)
        del pair_freqs[best_pair]

        a, b = best_pair
        new_token = a + b
        vocab[token_id] = new_token
        token_id += 1

        affected_words = []
        for word in word_freqs:
            bytes_list = word_bytes[word]
            for i in range(len(bytes_list) - 1):
                if bytes_list[i] == a and bytes_list[i + 1] == b:
                    affected_words.append(word)

        for word in affected_words:
            bytes_list = word_bytes[word]
            freq = word_freqs[word]

            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i + 1])
                if pair in pair_freqs:
                    pair_freqs[pair] -= freq
                    if pair_freqs[pair] <= 0:
                        del pair_freqs[pair]

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
                pair_freqs[pair] += freq
    pbar.close()
    print(f"BPE training completed. Final vocab size: {len(vocab)}, total merges: {len(merges)}.")
    return vocab, merges


def test_train_bpe():
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


if __name__ == "__main__":
    # test_train_bpe()
    import os
    import json

    tiny_input_path = os.path.expanduser("~/github/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    vocab, merges = train_bpe(
        tiny_input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    vocab_str = {idx: token.decode("utf-8", errors="ignore") for idx, token in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    
    with open("merges.txt", "w", encoding="utf-8") as f:
        for sub1, sub2 in merges:
            sub1 = sub1.decode("utf-8", errors="ignore")
            sub2 = sub2.decode("utf-8", errors="ignore")
            f.write(f"{sub1} {sub2}\n")
