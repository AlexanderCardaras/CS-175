from pathlib import Path

from nltk.corpus import cmudict
import markovify

import re
import string



def getsentences(the_text) :
    sents = re.findall(r"[A-Z].*?[\.!?]", the_text, re.M | re.DOTALL)
#    sents_lo = re.findall(r"[a-z].*?[\.!?]", the_text)
    return sents


def getwords(sentence) :
    x = re.sub('['+string.punctuation+']', '', sentence).split()
    return x
    
def sylco(word) :
    word = word.lower()
    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables
    
    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']
    
    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']
    
    pre_one = ['preach']

    syls = 0 #added syllable number
    disc = 0 #discarded syllable number

    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls
    
    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1
    
    #3) discard trailing "e", except where ending is "le"  
   
    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']
    
    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass
        
        else :
            disc+=1
    
    #4) check if consecutive vowels exists, triplets or pairs, count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple
    
    #5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))

    #6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1
        
    #7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1
        
    #8) add one if "y" is surrounded by non-vowels and is not in the last word.
    
    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1
    
    
    #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
    
    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1
    
    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1
    
    #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
    
    if word[-3:] == "ian" : 
    #and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1
    
    #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.
    
    if word[:2] == "co" and word[2] in 'eaoui' :
    
        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1

    #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1

    #13) check for "-n't" and cross match with dictionary to add syllable.
    
    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]
    
    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass   

    #14) Handling the exceptional words.
   
    if word in exception_del :
        disc+=1
        
    if word in exception_add :
        syls+=1     
    
    # calculate the output
    return numVowels - disc + syls    
    

# Get dictionary from cmudict
CMUDICT = cmudict.dict()

# Order of markov chain
N = 2

# Load compiled markov model
# If this code failed, run build_model.py first
with open('markov_chain/compiled_' + str(N) + '_order_model.json', 'r') as f:
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

def get_last_words(line):
  """
  Return the last n words of the given line, where n is the state_size of
  markov model (order of markov chain)
  :param line: a string with space-separated words
  :return: a string of last n words separated by a space
  """
  words = line.split()
  # Extract last state_size words
  last_token = words[-text_model.state_size:]
  return ' '.join(last_token)

def gen_n_syllable_sentence(n, init_words=None, must_start_with=False):
  """
  Generate an n-syllable sentence using Markov Chain model. If init_words is not
  provided, the sentence generation is random. If must_start_with is True,
  the returned sentence starts with init_words; otherwise, the returned sentence
  will be generated based on init_words but it's not included.
  If it cannot generate a sentence, KeyError exception will be raised.
  :param init_words: a string of the beginning word
  :param must_start_with: a bool indicating whether the returned sentence starts
    with init_words
  :return: a string of words with n syllables separated by a space
  """
  # If init_words must be at the beginning of the returned sentence, check the
  # syllable counts then (1) return if it fits (2) raise KeyError if it exceeds
  # or (3) continue generating the rest of the sentence
  if init_words is not None and must_start_with == True:
    if count_syllables_in_line(init_words) == n:
      return init_words
    elif count_syllables_in_line(init_words) > n:
      raise KeyError

  if init_words is None:
    s = text_model.make_sentence(max_words=n)
  elif must_start_with == True:
    s = text_model.make_sentence_with_start(init_words, strict=False, max_words=n)
  else:
    # Generate at most (n + state_size) words and strip init_words out
    s = text_model.make_sentence_with_start(init_words, strict=False, max_words=n+text_model.state_size)
    if s is not None:
      s = s[len(init_words)+1:]

  if s is not None and count_syllables_in_line(s) == n:
    return s.rstrip('.')

  raise KeyError

def gen_line_1(init_words=None):
  """
  Generate a 5-syllable sentence. If init_words is provided, the returned
  sentence will start with init_words. If it cannot generate a sentence,
  KeyError exception will be raised.
  :param init_words: a string of the beginning word
  :return: a string of words with 5 syllables separated by a space
  """
  if init_words is None:
    return gen_n_syllable_sentence(5)
  else:
    return gen_n_syllable_sentence(5, init_words=init_words, must_start_with=True)

def gen_line_2(init_words=None):
  """
  Generate a 7-syllable sentence. If init_words is provided, the returned
  sentence will be based on init_words. If it cannot generate a sentence,
  KeyError exception will be raised.
  :param init_words: a string of last words from the last line
  :return: a string of words with 7 syllables separated by a space
  """
  if init_words is None:
    return gen_n_syllable_sentence(7)
  else:
    return gen_n_syllable_sentence(7, init_words=init_words)

def gen_line_3(init_words=None):
  """
  Generate a 5-syllable sentence. If init_words is provided, the returned
  sentence will be based on init_words. If it cannot generate a sentence,
  KeyError exception will be raised.
  :param init_words: a string of last words from the last line
  :return: a string of words with 5 syllables separated by a space
  """
  if init_words is None:
    return gen_n_syllable_sentence(5)
  else:
    return gen_n_syllable_sentence(5, init_words=init_words)

def gen_haiku(init_words=None, independent_lines=False, tries=200):
  """
  Generate a full haiku with 3 lines (5-7-5 syllables structure) using Markov
  Chain model. If init_words is provided, the first line of the returned haiku
  will start with init_words. If it cannot generate a haiku, return None.
  :param init_words: a string of the beginning word
  :param independent_lines: True if each line should not be based on each other,
    otherwise False
  :param tries: a number of attempts to generate a sentence
  :return: a tuple of lines or None
  """
  # First line
  for _ in range(tries):
    try:
      s1 = gen_line_1(init_words)
      s1_end = get_last_words(s1)
    except KeyError:
      continue

    # Second line
    for _ in range(tries):
      try:
        s2 = gen_line_2(None if independent_lines else s1_end)
        s2_end = get_last_words(s2)
      except KeyError:
        continue

      # Third line
      for _ in range(tries):
        try:
          s3 = gen_line_3(None if independent_lines else s2_end)
          return s1,s2,s3
        except KeyError:
          continue

  return None
