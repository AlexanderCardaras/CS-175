import rnn_gen
from rnn_gen import get_data_from_file, RNNModule
import torch
from argparse import Namespace
from sylco import sylco
import numpy as np

int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file("parsed_data.txt", rnn_gen.flags.batch_size, rnn_gen.flags.seq_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = RNNModule(n_vocab, rnn_gen.flags.seq_size, rnn_gen.flags.embedding_size, rnn_gen.flags.lstm_size)
net.load_state_dict(torch.load('checkpoint_model/model-us-law-code.pth'))
net = net.to(device)

def get_next_word(words, output, syllable_number):
    current_syllables = 0
    for w in words:
        current_syllables = current_syllables + sylco(w)

    if current_syllables > syllable_number:
        return 0 # error: syllable count exceeded
    
    current_k = 5
    while current_k < 100:
        _, top_ix = torch.topk(output[0], k=current_k)
        list_of_possible_words = top_ix.tolist()[0]
        np.random.shuffle(list_of_possible_words)
        for possible_word in list_of_possible_words:
            choice_syllable_count = sylco(int_to_vocab[possible_word])

            if choice_syllable_count + current_syllables <= syllable_number:
                return int_to_vocab[possible_word]
        
        current_k = current_k + 5
    
    return 1 # error: generate new sentence

    
def predict_haiku_line(device, net, words, n_vocab, vocab_to_int, int_to_vocab, syllable_count=5):
    net.eval()
    # words = ['United', 'States'] # remove after training
    current_syllable = 0

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        try:
            ix = torch.tensor([[vocab_to_int[w]]]).to(device).long()
        except KeyError:
            ix = torch.tensor([[vocab_to_int['the']]]).to(device).long()
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        current_syllable = current_syllable + sylco(w)

    if current_syllable > syllable_count:
        print("Syllable count exceeded")
        return

    choice = get_next_word(words, output, syllable_count)
    if choice == 1:
        predict_haiku_line(device, net, words, n_vocab, vocab_to_int, int_to_vocab, syllable_count)
        return

    current_syllable = current_syllable + sylco(choice)
    words.append(choice)

    while current_syllable < syllable_count:
        ix = torch.tensor([[vocab_to_int[choice]]]).to(device).long()
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        choice = get_next_word(words, output, syllable_count)
        words.append(choice)
        current_syllable = current_syllable + sylco(choice)

    print(' '.join(words))

    ix = torch.tensor([[vocab_to_int[choice]]]).to(device).long()
    output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=5)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    return int_to_vocab[choice]


first_line = predict_haiku_line(device, net, ["The"], n_vocab, vocab_to_int, int_to_vocab, 5)
second_line = predict_haiku_line(device, net, [first_line], n_vocab, vocab_to_int, int_to_vocab, 7)
third_line = predict_haiku_line(device, net, [second_line], n_vocab, vocab_to_int, int_to_vocab, 5)


