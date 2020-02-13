import re
import bs4
import string
import nltk


def strip_json(file_path):
    """
    Strips lines of html and css. Converts file contents from dictionary to string format. Removes broken unicode
    :param file_path: Path to json file
    :return: List of strings decoded from html and css format
    """

    # List of laws
    contents = []

    # Open json file
    with open(file_path) as f:

        # Loop through each line in file
        for line in f:

            # Create Beautiful soup html data_parser object
            soup = bs4.BeautifulSoup(line, "html.parser")

            # Remove tags (soup.get_text) and json format (10:-4)
            stripped = str(soup.get_text()[10:-4])

            # Remove unicode that was decoded incorrectly (\\u201a, \\n, ...)
            stripped = re.sub(r'\\[\w][\d|\w][\d|\w][\d|\w][\d|\w]||\\n', '', stripped)

            # Remove things like
            #   section 3151(e)(2) of this title
            #   September 30, 2009,
            #   paragraph (1)
            #   subsection (a)
            #   subparagraph (A)(iv)
            #   clause (ii)
            #   (D)
            #   (1)
            stripped = re.sub(r"((\w+)\.?\s(\d+)\,\s(\d{4}))|"
                              r"(\w*(?<=section).*of this title)|"
                              r"(\w*(?<=sections).*of this title)|"
                              r"(\w*(?<=section).*of title \d+)|"
                              r"(\w*(?<=sections).*of title \d+)|"
                              r"(paragraph \(\d+\))|"
                              r"(subsection \(\w+\))|"
                              r"(subparagraph \(\w+\)\(\w+\))|"
                              r"(clause \(\w+\))|"
                              r"(\(\w+\))|"
                              r"(\(\d+\))"
                              , "", stripped)

            # Add lines that have text on them
            if len(stripped) > 0:
                contents.append(stripped)

    return contents


def list_to_file(data, file_path):
    """
    Writes a list of strings to a file
    :param data: List of strings
    :param file_path: Path to file
    :return: None
    """

    # Open target file
    with open(file_path, "w") as f:

        # Write each element of list to a new line
        for line in data:
            f.write(line + "\n")


def file_to_list(file_path):
    """
    Convert file contents into a list of strings
    :param file_path: Path to file
    :return: List of strings
    """

    # List of lines in file
    contents = []

    # Open file
    with open(file_path) as f:

        # Loop through each line in file
        for line in f:

            # Add line to list
            contents.append(line)

    return contents


def create_n_grams(data, n_grams=None, max_n_gram=7):
    """
    Splits data into n-grams, ranging from n=1 to n=max_n_gram
    :param data: List of strings
    :param n_grams: Dictionary of old n_grams, if avaliable
    :return: All n-grams found for each string element in data
    """

    # If n-gram dictionary doesnt exist yet
    if n_grams is None:

        # Initialize n-gram dictionary
        n_grams = {}
        for i in range(1, max_n_gram+1):
            n_grams[i] = []

    # Fill n-gram dictionary
    for line in data:

        # Change text to lowercase
        line = line.lower()

        # Remove punctuation
        line = line.translate(str.maketrans("", "", string.punctuation))

        # Split the line by space
        split = line.split()

        # Create n-grams ranging from n=1 to n=max_n_gram
        for n in range(1, max_n_gram+1):

            # Check if current sentence is long enough to support the n-gram
            if len(split) >= n:
                current_n_grams = list(nltk.ngrams(split, n))

                # Append each n-gram
                for n_gram in current_n_grams:
                    n_grams[n].append(n_gram)

            # else, sentence is too short for current n-gram and subsequent larger n-grams
            else:
                break

    return n_grams


def get_frequency(n_grams):
    """
    Finds the number of unique n-grams
    :param n_grams: Dictionary of lists of n-grams
    :return: Dictionary of lists of unique n-grams
    """

    # Unique n-grams
    n_gram_frequency = {}

    # Loop through all n-grams starting from 1-grams
    for i in range(1, len(n_grams)+1):

        # Create a dictionary for each layer (i) of n-grams
        n_gram_frequency[i] = {}

        # Loop through all the tokens in a layer and
        for token in n_grams[i]:

            # Add tokens to dictionary
            if token not in n_gram_frequency[i]:
                n_gram_frequency[i][token] = 1
            else:

                # Count number of times the specific token is repeated
                n_gram_frequency[i][token] = n_gram_frequency[i][token] + 1

    return n_gram_frequency
