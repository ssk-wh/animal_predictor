import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class Vocabulary:
    def __init__(self, min_freq=2):
        # Initialize vocabulary with special tokens
        self.word2idx = {'<pad>': PAD_TOKEN, '<sos>': SOS_TOKEN, '<eos>': EOS_TOKEN, '<unk>': UNK_TOKEN}
        self.idx2word = {PAD_TOKEN: '<pad>', SOS_TOKEN: '<sos>', EOS_TOKEN: '<eos>', UNK_TOKEN: '<unk>'}
        self.word_freq = {}
        self.min_freq = min_freq
        self.idx = 4  # Start indexing from 4
        
    def add_sentence(self, sentence):
        # Add words from a sentence to the vocabulary
        for word in word_tokenize(sentence.lower()):
            self.add_word(word)
            
    def add_word(self, word):
        # Add a single word to the vocabulary
        self.word_freq[word] = self.word_freq.get(word, 0) + 1
        if word not in self.word2idx and self.word_freq[word] >= self.min_freq:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __len__(self):
        return len(self.word2idx)
    
    def to_index(self, word):
        # Convert word to index, default to UNK_TOKEN if not found
        return self.word2idx.get(word, UNK_TOKEN)

class ChatDataset(Dataset):
    def __init__(self, pairs, vocab):
        # Initialize dataset with pairs of sentences and vocabulary
        self.pairs = pairs
        self.vocab = vocab
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # Retrieve a pair of sentences and convert them to tensors
        src, trg = self.pairs[idx]
        src_tensor = torch.LongTensor([self.vocab.to_index(word) for word in src])
        trg_tensor = torch.LongTensor([SOS_TOKEN] + [self.vocab.to_index(word) for word in trg] + [EOS_TOKEN])
        return src_tensor, trg_tensor

def collate_fn(batch):
    # Collate function for dataloader, pad sequences and return lengths
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    trg_lens = [len(x) for x in trg_batch]
    
    src_padded = pad_sequence(src_batch, padding_value=PAD_TOKEN, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_TOKEN, batch_first=True)
    
    return src_padded, trg_padded, src_lens, trg_lens

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        # Initialize embedding layer and LSTM
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lens):
        # Forward pass through encoder
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Initialize attention mechanism layers
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        # Compute attention weights
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask, -1e10)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        # Initialize decoder components including embedding, attention, and LSTM
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_TOKEN)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # Forward pass through decoder
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Attention computation
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        
        weighted = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_dim]
        
        # LSTM input
        lstm_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, emb_dim + hidden_dim]
        
        # Through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Final prediction
        output = torch.cat((output.squeeze(1), weighted.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        
        return prediction, hidden, cell, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        # Initialize sequence-to-sequence model with encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, src_lens, teacher_forcing_ratio=0.5):
        # Forward pass through the entire model
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        
        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        
        # Initial input (SOS token)
        input = trg[:, 0]
        
        # Create mask for padding
        mask = (src == PAD_TOKEN).to(device)
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get next token prediction
            top1 = output.argmax(1)
            
            # If using teacher forcing, next input comes from real data; otherwise from prediction
            input = trg[:, t] if teacher_force else top1
        
        return outputs

def prepare_data(file_path, min_freq=2):
    # Read data and create vocabulary and pairs of sentences
    df = pd.read_csv(file_path, usecols=['question', 'answer'])
    vocab = Vocabulary(min_freq=min_freq)
    
    print("Building vocabulary...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vocab.add_sentence(row['question'])
        vocab.add_sentence(row['answer'])
    
    # Convert to token sequences
    pairs = []
    for _, row in df.iterrows():
        q_tokens = word_tokenize(row['question'].lower())
        a_tokens = word_tokenize(row['answer'].lower())
        pairs.append((q_tokens, a_tokens))
    
    print(f"Vocabulary size: {len(vocab)}, Number of pairs: {len(pairs)}")
    return vocab, pairs

def train_model(model, dataloader, optimizer, criterion, n_epochs=10):
    # Train the model for a specified number of epochs
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for src, trg, src_lens, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, trg, src_lens)
            
            # Compute loss (ignoring SOS token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} Loss: {avg_loss:.4f}')
    
    # Save the model state and hyperparameters to a file
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size': len(vocab),
        'emb_dim': EMB_DIM,
        'hidden_dim': HID_DIM,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT
    }, 'model.pth')
    print("Model saved to model.pth")

def evaluate(model, vocab, sentence, max_length=30):
    # Evaluate the model on a single sentence and return the predicted response and attention weights
    model.eval()
    tokens = word_tokenize(sentence.lower())
    src_tensor = torch.LongTensor([vocab.to_index(word) for word in tokens]).unsqueeze(0).to(device)
    src_len = [len(tokens)]
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    trg_indexes = [SOS_TOKEN]
    attentions = []
    
    # Create mask for padding tokens in source sentence
    mask = (src_tensor == PAD_TOKEN).to(device)
    
    for _ in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell, attention = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs, mask)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        attentions.append(attention.cpu().numpy())
        
        if pred_token == EOS_TOKEN:
            break
    
    trg_tokens = [vocab.idx2word.get(idx, '<unk>') for idx in trg_indexes[1:-1]]
    return ' '.join(trg_tokens), np.array(attentions)

def test_model(model, vocab, test_data, num_examples=5):
    # Test the model on a few examples from the test data and print results
    model.eval()
    indices = random.sample(range(len(test_data)), min(num_examples, len(test_data)))
    
    for i in indices:
        src, trg = test_data[i]
        question = ' '.join(src)
        real_answer = ' '.join(trg)
        
        pred_answer, _ = evaluate(model, vocab, question)
        
        print(f"\n[Example {i+1}]")
        print(f"Q: {question}")
        print(f"Real Answer: {real_answer}")
        print(f"Predicted Answer: {pred_answer}")

def chat(model, vocab):
    # Interactive chat mode where user can input sentences and get responses from the model
    print("Type 'quit' to exit conversation")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response, _ = evaluate(model, vocab, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chatbot (PyTorch 2.2+ version)')
    parser.add_argument('mode', choices=['train', 'test', 'chat'], help='Run mode')
    parser.add_argument('--data', default='./data/conversations.csv', help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency of words')
    args = parser.parse_args()

    # Hyperparameters (adjustable)
    EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5

    vocab, pairs = prepare_data(args.data, min_freq=args.min_freq)

    if args.mode == 'train':
        print("Preparing data...")
        
        # Create dataset and dataloader
        dataset = ChatDataset(pairs, vocab)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch,
            collate_fn=collate_fn,
            shuffle=True
        )
        
        # Initialize model
        encoder = Encoder(len(vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
        decoder = Decoder(len(vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
        model = Seq2Seq(encoder, decoder).to(device)
        
        # Optimizer and loss function
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        
        print("Starting training...")
        train_model(model, dataloader, optimizer, criterion, args.epochs)
    
    elif args.mode in ['test', 'chat']:
        # Reload vocabulary
        _, pairs = prepare_data(args.data)
        
        # Load model
        checkpoint = torch.load('model.pth', map_location=device)
        encoder = Encoder(
            checkpoint['vocab_size'],
            checkpoint['emb_dim'],
            checkpoint['hidden_dim'],
            checkpoint['n_layers'],
            checkpoint['dropout']
        ).to(device)
        decoder = Decoder(
            checkpoint['vocab_size'],
            checkpoint['emb_dim'],
            checkpoint['hidden_dim'],
            checkpoint['n_layers'],
            checkpoint['dropout']
        ).to(device)
        model = Seq2Seq(encoder, decoder).to(device)
        model.load_state_dict(checkpoint['model_state'])
        
        if args.mode == 'test':
            test_model(model, vocab, pairs)
        else:
            chat(model, vocab)