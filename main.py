import json
import torch
import torch.nn as nn
from nltk_utils import tokenize, stem

# Define a simple RNN model
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

# Chatbot class to handle user interaction
class Chatbot:
    def __init__(self, model_path, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.model = SimpleRNN(len(self.vocab), 128, 256)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, question):
        question_indices = self.sentence_to_indices(question)
        
        input_ids = torch.tensor([question_indices]).to(self.device)
        
        with torch.no_grad():
            start_logits, end_logits = self.model(input_ids)
        
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        
        answer_tokens = input_ids[0][start_index:end_index + 1]
        answer = ' '.join([self.idx_to_token(idx.item()) for idx in answer_tokens])
        
        return answer

    def sentence_to_indices(self, sentence):
        tokens = tokenize(sentence)
        indices = [self.vocab.get(stem(token), self.vocab['<unk>']) for token in tokens]
        if len(indices) < 384:
            indices += [self.vocab['<pad>']] * (384 - len(indices))
        else:
            indices = indices[:384]
        return indices

    def idx_to_token(self, idx):
        for token, token_idx in self.vocab.items():
            if token_idx == idx:
                return token
        return '<unk>'

    def run(self):
        print("Hello! I am Vergil, your chatbot.")
        while True:
            question = input("you: ")
            if question.lower() in ['exit', 'quit', 'q']:
                print("vergil: Goodbye!")
                break
            answer = self.predict(question)
            print(f"vergil: {answer}")

if __name__ == "__main__":
    chatbot = Chatbot(model_path='model.pth', vocab_path='vocab.json')
    chatbot.run()
