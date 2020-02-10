import nltk
from data_parser import tokenizer
from nltk.corpus import cmudict
import math
import random


"""
POS tag list:

CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
"""

syl_dic = cmudict.dict()


def get_parts_of_speech(text):
    """
    Creates a parts of speech dictionary for the words in 'text'
    :param text: String of words / sentence
    :return: Dictionary containing parts of speech and word pairs
    """

    # Get all 1-grams in the sentence
    tokens = tokenizer.create_n_grams([text], max_n_gram=1)[1]

    # Convert from list of tuples to list of words
    tokens = [''.join(word) for word in tokens]

    # Find parts of speech for each token
    wp = nltk.pos_tag(tokens)

    # Group tokens by their parts of speech
    pos = {}
    for w, p in wp:
        if p in pos:
            pos[p].append(w)
        else:
            pos[p] = [w]

    return pos


def find_groupings():
    groupings = []

    return groupings


def combine_pos(pos, targets):
    final_list = []

    for target in targets:
        try:
            final_list += pos[target]
        except KeyError:
            continue

    return final_list


def token_syllable_count(token):
    """
    Finds the number of syllables in a token
    :param token: A single word
    :return: Number of syllables in token, or infinity if token is outside of the nltk dictionary
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in syl_dic[token.lower()]]
    except KeyError:
        return math.inf


def tokens_syllable_count(tokens):
    syllable_counts = []

    for token in tokens:
        syllable_counts.append((token, token_syllable_count(token)[0]))

    return syllable_counts


def get_pairs(list1, list2, max_syllable_count=5):
    pairs = []

    for l1 in list1:
        for l2 in list2:
            total_syllable_count = l1[1] + l2[1]
            if total_syllable_count <= max_syllable_count:
                pairs.append((l1[0], l2[0], total_syllable_count))

    return sorted(pairs, key=lambda x: x[2], reverse=True)


def construct_first_idea(text):
    """
    Sets the tone of the haiku by describing the scene in the first sentence
    :param pos: words categorized by their parts of speech
    :return: String of words with a syllable count of 5
    """

    pos = get_parts_of_speech(text)
    nouns = tokens_syllable_count(combine_pos(pos, ["NN", "NNS", "NNP", "NNPS"]))
    verbs = tokens_syllable_count(combine_pos(pos, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))

    pairs = get_pairs(verbs, nouns, max_syllable_count=5)

    return pairs[0][0] + " " + pairs[0][1]


def construct_second_idea(text):
    """
    Sets the tone of the haiku by describing the scene in the first sentence
    :param pos: words categorized by their parts of speech
    :return: String of words with a syllable count of 5
    """

    pos = get_parts_of_speech(text)

    nouns = tokens_syllable_count(combine_pos(pos, ["NN", "NNS", "NNP", "NNPS"]))
    verbs = tokens_syllable_count(combine_pos(pos, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))

    pairs = get_pairs(verbs, nouns, max_syllable_count=7)

    return pairs[0][0] + " " + pairs[0][1]


# input1 = "The term deal in includes making, taking, buying, selling, redeeming, or collecting."
# input2 = "People all around."
#
# s1 = construct_first_idea(input1)
# s2 = construct_second_idea(input2)
#
# print([len(list(y for y in x if y[-1].isdigit())) for x in syl_dic["federal"]])
#
#
# print(s1)
# print(s2)
