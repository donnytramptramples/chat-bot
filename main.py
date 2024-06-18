import json
import torch
from model import Encoder, Attention, Decoder
from nltk_utils import tokenize, stem, sentence_to_indices

def generate_response(encoder, decoder, input_sentence, vocab, device, max_length=50):
    input_indices = sentence_to_indices(input_sentence, vocab, max_length)
    input_tensor = torch.tensor([input_indices]).to(device)
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[0]], device=device)  # Start token index
        
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if topi.item() == 1:  # End token
                break

            decoded_words.append(vocab[topi.item()])

        output_sentence = ' '.join(decoded_words)
    
    return output_sentence

if __name__ == "__main__":
    print("Loading the model and vocabulary for interaction...")
    
    # Load vocabulary from JSON file
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # Initialize the encoder and decoder using the loaded vocabulary size
    embed_size = 128
    hidden_size = 256
    encoder = Encoder(len(vocab), embed_size, hidden_size)
    attention = Attention(hidden_size)
    decoder = Decoder(hidden_size, len(vocab), attention)

    # Move encoder and decoder to appropriate device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
    encoder.to(device)
    decoder.to(device)
    
    print("Chatbot is ready to interact! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(encoder, decoder, user_input, vocab, device)
        print(f"Bot: {response}")
