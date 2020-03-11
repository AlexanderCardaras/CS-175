from src.markov_chain.markov_chain import gen_haiku
from src.rnn import generate_text

"""
Generate 5 random haikus
"""

if __name__ == '__main__':
  while True:
    s = gen_haiku()
    if s is not None:
      print('\n'.join(s))
      break

  print("")
  generate_text.call_me()
