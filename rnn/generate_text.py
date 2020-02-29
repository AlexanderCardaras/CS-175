import rnn_gen
from rnn_gen import get_data_from_file, RNNModule
import torch
from argparse import Namespace
from nltk.corpus import cmudict
from sylco import sylco
import numpy as np
import random
import string
import pickle

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.heuristics import classify

int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file("parsed_data.txt", rnn_gen.flags.batch_size, rnn_gen.flags.seq_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open ('get_heuristic_model.p', 'rb') as f:
    heuristic_model = pickle.load(f)

net = RNNModule(n_vocab, rnn_gen.flags.seq_size, rnn_gen.flags.embedding_size, rnn_gen.flags.lstm_size)
net.load_state_dict(torch.load('checkpoint_model/model-us-law-code.pth'))
net = net.to(device)
CMUDICT = cmudict.dict()

def count_syllables_in_word(word):
  """
  Return the number of syllables of the word by based on nltk pronunciation
  dictionary and sylco (heuristic-based syllable counter)
  :param word: a string
  :return: an integer of syllable counts
  """
  try:
    return [len(list(y for y in x if y[-1].isdigit())) for x in CMUDICT[word.lower()]][0]
  except KeyError:
    return sylco(word)

def get_next_word(words, output, syllable_number):
    current_syllables = 0
    for w in words:
        current_syllables = current_syllables + count_syllables_in_word(w)

    if current_syllables > syllable_number:
        return 0 # error: syllable count exceeded
    
    current_k = 5
    while current_k < 100:
        _, top_ix = torch.topk(output[0], k=current_k)
        list_of_possible_words = [item for item in top_ix.tolist()[0] if count_syllables_in_word(int_to_vocab[item]) <= syllable_number - current_syllables]
        np.random.shuffle(list_of_possible_words)
        for possible_word in [item for item in list_of_possible_words if int_to_vocab[item] not in string.punctuation]:
            choice_syllable_count = count_syllables_in_word(int_to_vocab[possible_word])

            if choice_syllable_count + current_syllables < syllable_number:
                return int_to_vocab[possible_word]
            elif choice_syllable_count + current_syllables == syllable_number:
                if classify(int_to_vocab[possible_word], heuristic_model):
                    return int_to_vocab[possible_word]

        current_k = current_k + 5
    
    return None # error: generate new sentence

    
def predict_haiku_line(device, net, words, n_vocab, vocab_to_int, int_to_vocab, syllable_count=5, h=None, c=None):
    net.eval()
    # words = ['United', 'States'] # remove after training
    current_syllable = 0

    if h == None and c == None:
        state_h, state_c = net.zero_state(1)
    else:
        state_h = h
        state_c = c
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        try:
            ix = torch.tensor([[vocab_to_int[w]]]).to(device).long()
        except KeyError:
            ix = torch.tensor([[vocab_to_int['the']]]).to(device).long()
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        current_syllable = current_syllable + count_syllables_in_word(w)

    if current_syllable > syllable_count:
        print("Syllable count exceeded")
        return

    choice = get_next_word(words, output, syllable_count)
    # if choice == 1:
    #     predict_haiku_line(device, net, words, n_vocab, vocab_to_int, int_to_vocab, syllable_count, h, c)
    #     return

    current_syllable = current_syllable + count_syllables_in_word(choice)
    words.append(choice)

    while current_syllable < syllable_count:
        ix = torch.tensor([[vocab_to_int[choice]]]).to(device).long()
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        choice = get_next_word(words, output, syllable_count)
        words.append(choice)
        current_syllable = current_syllable + count_syllables_in_word(choice)

    haiku_line = ' '.join(words)
    print(haiku_line.translate(str.maketrans('', '', string.punctuation)))

    ix = torch.tensor([[vocab_to_int[choice]]]).to(device).long()
    output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=5)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    return (int_to_vocab[choice], state_h, state_c)

for x in range(0, 10):
    first_line = predict_haiku_line(device, net, ['Such', 'court'], n_vocab, vocab_to_int, int_to_vocab, 5)
    second_line = predict_haiku_line(device, net, [first_line[0]], n_vocab, vocab_to_int, int_to_vocab, 7, first_line[1], first_line[2])
    third_line = predict_haiku_line(device, net, [second_line[0]], n_vocab, vocab_to_int, int_to_vocab, 5, second_line[1], second_line[2])
    print(" ")


