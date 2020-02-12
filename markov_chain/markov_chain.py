from pathlib import Path

from nltk.corpus import cmudict
import markovify

from sylco import sylco

# Get dictionary from cmudict
CMUDICT = cmudict.dict()

# Load compiled markov model
# If this code failed, run build_model.py first
with open('markov_chain/compiled_model.json', 'r') as f:
  text_model = markovify.Text.from_json(f.read())

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

def count_syllables_in_line(line):
  """
  Return the total number of syllables in a line
  :param line: a string with space-separated words
  :return: an integer of total syllable counts
  """
  ws = line.rstrip('.').split()
  return sum([count_syllables_in_word(w) for w in ws])

def gen_5_syllable_sentence():
  """
  Generate a 5-syllable sentence using Markov Chain model
  :return: a string of words with 5 syllables
  """
  while True:
    s = text_model.make_sentence(max_words=4)
    if s is not None and count_syllables_in_line(s) == 5:
      return s.rstrip('.')

def gen_7_syllable_sentence():
  """
  Generate a 7-syllable sentence using Markov Chain model
  :return: a string of words with 7 syllables
  """
  while True:
    s = text_model.make_sentence(max_words=6)
    if s is not None and count_syllables_in_line(s) == 7:
      return s.rstrip('.')

def markov_gen():
  """
  Generate a full haiku with 3 lines (5-7-5 syllables structure) using Markov
  Chain model
  :return: a tuple of lines
  """
  s1 = gen_5_syllable_sentence()
  s2 = gen_7_syllable_sentence()
  s3 = gen_5_syllable_sentence()
  return s1,s2,s3
