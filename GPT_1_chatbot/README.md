
## Files and Functions

### loader.py
Main file for running the chatbot. Handles model inference and interactive conversation.

Key functions:
- `encode(text)`: Converts text to token IDs using the trained tokenizer
- `decode(token_ids)`: Converts token IDs back to readable text
- `chat(prompt, history=None, max_new_tokens=60, temperature=0.7, top_k=40, min_new_tokens=3, history_max_tokens=None)`: Main chat function that generates replies while maintaining conversation history. Manages token limits and prevents premature generation termination.
- `GPT.generate()`: Generates new tokens autoregressively with configurable sampling strategies including temperature scaling and top-k filtering
- `GPT.forward()`: Computes logits and optional loss for training/inference

Model architecture components:
- `Head`: Single attention head with query, key, value projections
- `MultiHeadAttention`: Multiple attention heads combined with projection
- `FeedForward`: Feed-forward network with ReLU activation
- `Block`: Transformer block combining multi-head attention and feed-forward with layer normalization and residual connections
- `GPT`: Full model combining embeddings, transformer blocks, and language modeling head

### train.py
Handles model training pipeline.

Key functions:
- `get_batch(split)`: Creates random batches of training or validation data
- `estimate_loss()`: Evaluates model performance on both training and validation sets

Model hyperparameters:
- block_size: 384 tokens context window
- batch_size: 48 parallel sequences
- n_embed: 512 embedding dimension
- n_layers: 6 transformer blocks
- n_head: 8 attention heads
- epochs: 5000 training iterations
- dropout: 0.15

### token_generator.py
Generates custom tokenizer and processes the training dataset.

Key functions:
- `train_tokenizer()`: Trains a GPT2-based tokenizer on the TinyChat dataset with 16,384 vocabulary size. Adds special tokens for user/assistant roles and EOS marker.
- `convert_inst_to_roles(text)`: Splits multi-turn conversations into individual user-assistant pairs with proper formatting
- `chunk_conversation(text, tokenizer, block_size)`: Chunks conversation text into segments respecting the token limit
- `process_and_tokenize_dataset(tokenizer)`: Processes entire dataset, converts conversations to token sequences, and appends EOS tokens
- `save_tokenized_data(token_ids, chunks, tokenizer)`: Saves tokenized data, chunks, and metadata to disk for efficient training

## Usage

Run the interactive chatbot:

```
python loader.py
```

The chatbot will load the trained model and tokenizer, then enter an interactive chat loop. Type your messages and press Enter. Type 'quit' or 'exit' to end the conversation. 