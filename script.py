from markov_chain.markov_chain import gen_haiku
import os
import subprocess
"""
Generate 5 random haikus
"""

if __name__ == '__main__':
  while True:
    s = gen_haiku()
    if s is not None:
      print('\n'.join(s))
      break
  subprocess.check_call(['python', 'generate_text.py'], cwd='./rnn/')