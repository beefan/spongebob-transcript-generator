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

  #Instantiate the network
  net = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
  net = net.to(device)

  criterion, optimizer = get_loss_and_train_op(net, 0.01)

  iteration = 0
  for e in range(50):
    batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
    state_h, state_c = net.zero_state(flags.batch_size)
     
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for x, y in batches:
        iteration += 1
        
        # train
        net.train()
        
        # reset gradients
        optimizer.zero_grad()
        
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
            
        logits, (state_h, state_c) = net(x, (state_h, state_c))
        loss = criterion(logits.transpose(1,2), y)
        
        # detach() so pytorch can calculate loss
        state_h = state_h.detach()
        state_c = state_c.detach()
        
        loss_value = loss.item()
        
        # back propagation
        loss.backward()
        
        # gradient clipping 
        _ = torch.nn.utils.clip_grad_norm_(net.parameters(), flags.gradients_norm)
        
        # update network parameters
        optimizer.step()
        
        # print loss values to the console during training
        if iteration % 100 == 0:
            print('Epoch: {}/{}'.format(e, 200),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss_value))
        
        # print a little sample of text during training
        if iteration % 1000 == 0:
            predict(device, net, flags.initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5)
            torch.save(net.state_dict(), '../checkpoint_pt/model-{}-{}.pth'.format(iteration, loss_value))

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

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

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
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])
    
    print(' '.join(words))

main()