from markov_chain import gen_haiku

"""
Generate 5 random haikus
"""

if __name__ == '__main__':
  for i in range(5):
    while True:
      s = gen_haiku()
      if s is not None:
        print('\n'.join(s))
        break
    print('=' * 20)
