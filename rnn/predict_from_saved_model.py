import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from argparse import Namespace

flags = Namespace(
    train_file='../web_scraper/spongebob-transcript.txt',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['_________________________________________________________'],
    predict_top_k=5,
    checkpoint_path='checkpoint'
)

def main():
  device = torch.device('cpu')
  int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = process_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)
  
  model = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
  model.load_state_dict(torch.load('../checkpoint_pt/model-9000-4.580765247344971.pth'))
  model.eval()

  predict(device, model, flags.initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5)

def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    # tells the network we are about to evaluate
    net.eval()
    
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    
    # append another word
    words.append(int_to_vocab[choice])
    
    # append 100 more
    for _ in range(4000):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])
    
    print(' '.join(words))

def process_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r') as file:
        text = file.read()
    text = text.split()
    
    word_counts = Counter(text)
    
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    
    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

class RNNModule(nn.Module):
    #define necessary layers in the constructor:
    #embedding layer, LSTM layer, dense layer
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
    
    #define function for forward pass
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state
    
    #define function to reset state at each epoch
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

main()
