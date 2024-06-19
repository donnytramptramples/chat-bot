import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)  # Apply relu activation
        output, hidden = self.gru(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # Ensure hidden is of shape (batch_size, hidden_size)
        hidden = hidden.unsqueeze(1)  # shape: (batch_size, 1, hidden_size)

        # Ensure encoder_outputs is of shape (batch_size, max_len, hidden_size)
        # This assumes encoder_outputs is already (batch_size, max_len, hidden_size)
        # If needed, you might have to permute dimensions or reshape encoder_outputs
        attn_weights = torch.bmm(encoder_outputs, hidden.transpose(1, 2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(encoder_outputs.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(input.size(0), 1, -1)
        embedded = F.relu(embedded)  # Apply relu activation
        context = self.attention(hidden[-1], encoder_outputs)
        combined = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(combined, hidden)
        output = F.log_softmax(self.out(torch.cat((output, context.unsqueeze(1)), dim=2)), dim=2)
        return output, hidden, context
