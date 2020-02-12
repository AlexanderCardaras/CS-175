from markov_chain import markov_gen

"""
Generate 5 random haikus
"""

if __name__ == '__main__':
  for i in range(5):
    print('\n'.join(markov_gen()))
    print('=' * 20)
