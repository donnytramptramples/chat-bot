# train.py

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from nltk_utils import build_vocab, sentence_to_indices
from model import Encoder, Attention, Decoder

# Example data loading function, adjust as per your dialog.txt format
def load_dialogs(filename):
    contexts = []
    questions = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            context, question = line.strip().split('\t')
            contexts.append(context)
            questions.append(question)
    return contexts, questions

# Define your custom dataset class
class ConversationDataset(Dataset):
    def __init__(self, filename):
        self.contexts, self.questions = load_dialogs(filename)
        self.vocab = build_vocab(self.contexts + self.questions)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        input_indices = sentence_to_indices(context, self.vocab, max_length=50)
        target_indices = sentence_to_indices(question, self.vocab, max_length=50)
        return torch.tensor(input_indices), torch.tensor(target_indices)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embed_size = 256
hidden_size = 512
learning_rate = 0.001
epochs = 10

# Initialize dataset and access its vocab attribute
train_dataset = ConversationDataset('dialog.txt')
vocab_size = len(train_dataset.vocab)

# Check if '<SOS>' token exists in vocab, otherwise handle the case
sos_index = train_dataset.vocab.get('<SOS>', None)
if sos_index is None:
    raise ValueError("'<SOS>' token not found in the vocabulary!")

# Initialize model components
encoder = Encoder(vocab_size, embed_size, hidden_size).to(device)
attention = Attention(hidden_size).to(device)
decoder = Decoder(hidden_size, vocab_size, attention).to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(attention.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Dummy DataLoader, replace with your actual DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

def train_model(encoder, decoder, train_loader, device):
    encoder.train()
    decoder.train()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for step, (input_ids, target_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            encoder_outputs, encoder_hidden = encoder(input_ids)
            decoder_hidden = encoder_hidden

            # Handle case where <SOS> token is not found
            decoder_input = torch.tensor([[sos_index]] * input_ids.size(0), device=device)
            if sos_index is None:
                raise ValueError("'<SOS>' token not found in the vocabulary!")

            max_target_length = target_ids.size(1)
            loss = 0
            for t in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output.squeeze(1), target_ids[:, t])
                decoder_input = target_ids[:, t].unsqueeze(1)

            total_loss += loss.item() / max_target_length
            loss.backward()
            optimizer.step()

            # Print debugging information
            print(f"Batch {step + 1}/{len(train_loader)}, Loss: {loss.item() / max_target_length}")

        print(f"Average Train Loss: {total_loss / len(train_loader)}")

train_model(encoder, decoder, train_loader, device)
