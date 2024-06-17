import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter
from nltk_utils import tokenize, stem

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc_start = nn.Linear(hidden_size, 1)
        self.fc_end = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embeds)
        start_logits = self.fc_start(rnn_out).squeeze(-1)
        end_logits = self.fc_end(rnn_out).squeeze(-1)
        return start_logits, end_logits

def load_squad_format_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    contexts = []
    questions = []
    answers = []

    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                
                if qa['answers']:
                    answer_text = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                else:
                    answer_text = ""
                    answer_start = -1
                
                is_impossible = qa.get('is_impossible', False)

                contexts.append(context)
                questions.append(question)
                answers.append({'text': answer_text, 'answer_start': answer_start, 'is_impossible': is_impossible})
    
    return contexts, questions, answers

def build_vocab(sentences, min_freq=5, max_size=None):
    word_counts = Counter()
    for sentence in sentences:
        tokens = tokenize(sentence)
        word_counts.update(tokens)
    
    filtered_vocab = {'<pad>': 0, '<unk>': 1}
    for i, (word, count) in enumerate(word_counts.items(), start=len(filtered_vocab)):
        if count >= min_freq:
            filtered_vocab[word] = i
        if max_size is not None and len(filtered_vocab) >= max_size:
            break
    
    return filtered_vocab

def sentence_to_indices(sentence, vocab, max_length):
    tokens = tokenize(sentence)
    indices = [vocab.get(stem(token), vocab['<unk>']) for token in tokens]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

def map_char_to_token(context, char_pos, vocab, max_length):
    tokens = tokenize(context)
    char_to_token = []
    token_start = 0
    for i, token in enumerate(tokens):
        token_end = token_start + len(token)
        for _ in range(token_start, token_end):
            char_to_token.append(i)
        token_start = token_end + 1  # +1 for space

    if char_pos >= len(char_to_token):
        return len(tokens)  # Return the length of tokens if char_pos is out of bounds
    return char_to_token[char_pos]

def train_squad_model(model, train_loader, device, epochs=3, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_batches = len(train_loader)
    total_steps = epochs * total_batches
    progress = tqdm(total=total_steps, desc='Training Progress', position=0)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            start_positions = batch[1].to(device)
            end_positions = batch[2].to(device)

            valid_indices = (start_positions != -1) & (end_positions != -1)
            if not valid_indices.any():
                continue

            input_ids = input_ids[valid_indices]
            start_positions = start_positions[valid_indices]
            end_positions = end_positions[valid_indices]

            # Debugging: Print maximum index value
            max_index = input_ids.max().item()
            if max_index >= model.embedding.num_embeddings:
                print(f"Error: max index {max_index} out of bounds for embedding layer with size {model.embedding.num_embeddings}")
                continue

            # Clamp the positions to ensure they are within the valid range
            start_positions = torch.clamp(start_positions, 0, input_ids.size(1) - 1)
            end_positions = torch.clamp(end_positions, 0, input_ids.size(1) - 1)

            start_logits, end_logits = model(input_ids)
            loss_start = criterion(start_logits, start_positions)
            loss_end = criterion(end_logits, end_positions)
            loss = (loss_start + loss_end) / 2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.update(1)
            progress.set_postfix({'Epoch': epoch + 1, 'Loss': total_loss / (step + 1)})

        avg_train_loss = total_loss / total_batches
        print(f"Average Train Loss: {avg_train_loss:.4f}")

    progress.close()
    print("Training complete.")

if __name__ == "__main__":
    squad_file = 'train-v2.0.json'
    train_contexts, train_questions, train_answers = load_squad_format_data(squad_file)
    all_sentences = train_contexts + train_questions
    vocab = build_vocab(all_sentences, min_freq=5, max_size=10000)
    vocab_size = len(vocab)

    max_length = 384
    context_indices = [sentence_to_indices(context, vocab, max_length) for context in train_contexts]
    question_indices = [sentence_to_indices(question, vocab, max_length) for question in train_questions]

    # Validate indices and filter out of bounds
    context_indices = [[idx if idx < vocab_size else vocab['<unk>'] for idx in indices] for indices in context_indices]
    question_indices = [[idx if idx < vocab_size else vocab['<unk>'] for idx in indices] for indices in question_indices]

    # Map answer start and end positions to token indices
    start_positions = [map_char_to_token(context, a['answer_start'], vocab, max_length) if a['answer_start'] != -1 else 0 for context, a in zip(train_contexts, train_answers)]
    end_positions = [map_char_to_token(context, a['answer_start'] + len(a['text']), vocab, max_length) if a['answer_start'] != -1 else 0 for context, a in zip(train_contexts, train_answers)]

    dataset = TensorDataset(torch.tensor(context_indices), torch.tensor(start_positions), torch.tensor(end_positions))
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    embed_size = 128
    hidden_size = 256
    model = SimpleRNN(vocab_size, embed_size, hidden_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_squad_model(model, train_loader, device)

    torch.save(model.state_dict(), 'model.pth')
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    print("Training and saving complete.")
