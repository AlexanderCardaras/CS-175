from collections import defaultdict

from markov_chain import text_model, count_syllables_in_line

"""
An experiment to find the proper max_words parameter by generating many
sentences using different max_words and choosing the max_words that have the
highest probability of forming a 5-syllable sentence and a 7-syllable sentence
"""

N = 1000

def formatted_counts(count):
  """
  Return a formatted string of counts and its percentage
  Example: count = 135, N = 1000
    returned = '135 = 13.50%'
  :param count: an integer of counts
  :return: a formatted string
  """
  return str(count) + ' = {0:.2f}%'.format(count/N*100) 

if __name__ == "__main__":
  # Loop over different max_words
  for max_words in range(3, 11):
    # key = number of syllables
    # value = number of generated sentences with that syllables
    syllable_count = defaultdict(int)

    # Generate N sentences 
    for i in range(N):
      s = text_model.make_sentence(tries=250, max_words=max_words)
      if s is None:
        # Add to 0 if it failed to generate a sentence
        syllable_count[0] += 1
      else:
        syllable_count[count_syllables_in_line(s)] += 1

    print('Max words:', max_words)
    print(sorted(syllable_count.items(), key=(lambda i: (i[1],i[0])), reverse=True))
    print('Failed:', formatted_counts(syllable_count[0]))
    print('5-syllable sentence:', formatted_counts(syllable_count[5]))
    print('7-syllable sentence:', formatted_counts(syllable_count[7]))
    print('=' * 20)
