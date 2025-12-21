import os
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset

from transformers import GPT2TokenizerFast
import pickle

# Utilize tokenizer algorithm from GPT-2
BLOCK_SIZE = 384  # context window
VOCAB_SIZE = 16384
OUTPUT_DIR = "./tokenizer"
DATA_OUTPUT_DIR = "./tokenizer/processed_data"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading TinyChat dataset...")
ds = load_dataset("starhopp3r/TinyChat")
print(f"Dataset loaded. Training samples: {len(ds['train'])}")
print(f"Sample: {ds['train'][0]}")

def convert_inst_to_roles(text):
    """Convert instruction format text to user/assistant role format."""
    chunks = []
    parts = text.split("[INST]")

    for part in parts:
        if "[/INST]" not in part:
            continue

        user, assistant = part.split("[/INST]", 1)

        user = user.strip()
        assistant = assistant.strip()

        if user:
            chunks.append(f"<|user|> {user}")
        if assistant:
            chunks.append(f"<|assistant|> {assistant}")

    return "\n".join(chunks)


def tokenizer_text_iterator():
    """Iterator that yields converted text from dataset."""
    for sample in ds["train"]:
        text = sample["text"]
        converted_text = convert_inst_to_roles(text)
        if converted_text.strip():
            yield converted_text


def chunk_conversation(text, tokenizer, block_size):
    """Chunk conversation text based on token limits."""
    lines = text.split("\n")
    chunks = []

    buffer = ""

    for line in lines:
        candidate = buffer + line + "\n"
        token_count = len(tokenizer(candidate)["input_ids"])

        if token_count > block_size:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = line + "\n"
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def train_tokenizer():
    """Train tokenizer on dataset."""
    print("\nTraining tokenizer...")
    base_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    tokenizer = base_tokenizer.train_new_from_iterator(
        tokenizer_text_iterator(),
        vocab_size=VOCAB_SIZE
    )

    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "eos_token": "<|eos|>",
        "additional_special_tokens": ["<|user|>", "<|assistant|>"]
    })

    print(f"Tokenizer trained with vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tokenizer saved to {OUTPUT_DIR}")
    
    return tokenizer


def process_and_tokenize_dataset(tokenizer):
    """Process entire dataset into chunks and tokenize."""
    print("\nProcessing and tokenizing dataset...")
    
    all_token_ids = []
    all_chunks = []
    chunk_count = 0
    
    for idx, sample in enumerate(ds["train"]):
        if idx % 100 == 0:
            print(f"Processing sample {idx}/{len(ds['train'])}")
        
        text = sample["text"]
        converted_text = convert_inst_to_roles(text)
        
        if not converted_text.strip():
            continue
        
        # Chunk the text
        chunks = chunk_conversation(converted_text, tokenizer, BLOCK_SIZE)
        
        for chunk in chunks:
            # Tokenize chunk
            tokens = tokenizer(
                chunk,
                return_tensors=None,
                truncation=True,
                max_length=BLOCK_SIZE + 1
            )["input_ids"]
            
            all_token_ids.append(tokens)
            all_chunks.append(chunk)
            chunk_count += 1
    
    print(f"\nTotal chunks created: {chunk_count}")
    
    return all_token_ids, all_chunks


def save_tokenized_data(token_ids, chunks):
    """Save tokenized data for training."""
    print("\nSaving tokenized data...")
    
    # Convert to numpy array for efficient storage
    token_array = np.array(token_ids, dtype=object)
    
    # Save token IDs
    token_file = os.path.join(DATA_OUTPUT_DIR, "token_ids.npy")
    np.save(token_file, token_array)
    print(f"Token IDs saved to {token_file}")
    
    # Save chunks as reference
    chunks_file = os.path.join(DATA_OUTPUT_DIR, "chunks.pkl")
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_file}")
    
    # Save metadata
    metadata = {
        "total_chunks": len(chunks),
        "block_size": BLOCK_SIZE,
        "vocab_size": VOCAB_SIZE,
        "avg_tokens_per_chunk": np.mean([len(t) for t in token_ids])
    }
    
    metadata_file = os.path.join(DATA_OUTPUT_DIR, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")
    print(f"Average tokens per chunk: {metadata['avg_tokens_per_chunk']:.2f}")


def main():
    """Main pipeline for tokenizer generation and data processing."""
    try:
        # Train tokenizer
        tokenizer = train_tokenizer()
        
        # Process and tokenize dataset
        token_ids, chunks = process_and_tokenize_dataset(tokenizer)
        
        # Save tokenized data
        save_tokenized_data(token_ids, chunks)
        
        print("\n✓ Tokenizer generation complete!")
        print(f"✓ Tokenizer saved to: {OUTPUT_DIR}")
        print(f"✓ Processed data saved to: {DATA_OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error during tokenizer generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()