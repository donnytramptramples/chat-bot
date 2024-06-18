# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, max_length=50):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.max_length = max_length

    def forward(self, input):
        embedded = self.embedding(input)
        
        # Pack the embedded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, [self.max_length] * input.size(0), batch_first=True, enforce_sorted=False)
        
        # Forward pass through GRU
        output, hidden = self.rnn(packed_embedded)
        
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)  # Removed multiplication by 2
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if isinstance(encoder_outputs, torch.nn.utils.rnn.PackedSequence):
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        
        max_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(hidden) + self.attn(encoder_outputs))  # Update energy calculation
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, max_length=50):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.max_length = max_length

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        
        # Ensure context is compatible with the encoder output
        context = self.attention(hidden[-1], encoder_outputs)
        
        # Concatenate embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Forward pass through GRU
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Predict next token with output of GRU
        output = self.out(torch.cat((output, context), dim=2))
        output = F.log_softmax(output, dim=2)
        
        return output, hidden, context
