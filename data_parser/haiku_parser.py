from data_parser import tokenizer
import pandas as pd
import nltk
import numpy as np


def open_csv(file_path, sample_size=0.01):
    """
    Uses pandas to open csv file
    :param file_path: Path to csv file
    :param sample_size: Sample size [0-1] of data we will take from file
    :return: A fraction of the contents of the file equal to sample_size
    """
    contents = pd.read_csv(file_path)
    return contents.sample(frac=sample_size)


# TODO: Add Comments, make code more readable
def get_poems(path, sample_size=0.1):
    """
    Barebones version of "Format Info For training"
        (https://github.com/docmarionum1/haikurnn/blob/master/notebooks/models/v1/Training.ipynb)
    Specialty function for reading and parsing haiku.csv
    :param path: Path to haiku.csv file
    :param sample_size:
    :return: [Lists of poems indexed by line], [List of syllable counts for each line of each poem]
    """

    df = open_csv(path, sample_size)

    lines = set([0, 1, 2])

    for i in range(3):
        lines.remove(i)
        df = df[['0', '1', '2',] + ['%s_syllables' % j for j in lines]].join(
            df['%s_syllables' % i].str.split(',', expand=True).
                stack(-1).reset_index(level=1, drop=True).rename('%s_syllables' % i)
        ).drop_duplicates()
        lines.add(i)

    for i in range(3):
        df['%s_in' % i] = (df[str(i)])

        if i == 2:  # If it's the last line
            df['%s_out' % i] = df[str(i)]
        else:
            # If it's the first or second line, add the first character of the next line to the end of this line.
            # This helps with training so that the next RNN has a better chance of getting the first character right.
            df['%s_out' % i] = (df[str(i)] + df[str(i + 1)].str[0])

    inputs = df[['0_in', '1_in', '2_in']].values
    syllables = df[['0_syllables', '1_syllables', '2_syllables']].values

    return inputs, syllables


def filter_poems(poems, syllables, target=('5', '7', '5')):
    """
    Finds poems with target syllable count
    :param poems: List of poems
    :param syllables: List of syllable counts for poems
    :param target: Target syllable count
    :return: New list of poems and syllable counts that contain only the target syllable count
    """

    # Find the indices of poems that match target syllable count
    indices = np.where((syllables == target).all(axis=1))

    # Get target poems
    new_poems = poems[indices]
    new_syllables = syllables[indices]

    return new_poems, new_syllables


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
