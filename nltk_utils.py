import nltk
import torch
from collections import Counter

# Function to build vocabulary
def build_vocab(data):
    all_words = []
    for sentence in data:
        all_words.extend(sentence.split())
    word_counts = Counter(all_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

    # Insert special tokens in the vocabulary
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}  # Define special tokens
    vocab.update({word: idx + 4 for idx, word in enumerate(sorted_words)})  # Start index for words

    return vocab

# Function to convert sentence to indices based on vocabulary
def sentence_to_indices(sentence, vocab, max_length):
    tokenized = nltk.word_tokenize(sentence.lower())
    indices = [vocab[token] if token in vocab else vocab['<UNK>'] for token in tokenized]
    indices = indices[:max_length] + [vocab['<PAD>']] * (max_length - len(indices))
    return indices