


def open_file(file_path):
    """ opens a text file for reading """

    contents = None
    f = open(file_path, 'r')

    # check to see that file was successfully opened
    if f.mode == 'r':
        contents = f.read()

    return contents


def tokenize(text):
    """
    Open text file and create a list of all words
    :param text: Text file
    :return: List of tokens
    """
    """ """

    # change text to lowercase
    text = text.lower()

    # deliminate by special characters inspiration from
    # https://stackoverflow.com/questions/1276764/stripping-everything-but-alphanumeric-chars-from-a-string-in-python
    text = re.sub(r'\W+', ' ', text)
    my_tokens = re.split(' ', text)

    # array for storing tokens
    all_tokens = []

    # add all tokens to the tokens list
    for token in my_tokens:

        # in some cases r.split will return an empty string as a token
        # one case is if two delimiting characters are next to each other
        if len(token) > 0:
            all_tokens.append(token)

    return all_tokens



def compute_word_frequencies(my_tokens):
    """

    :param my_tokens:
    :return:
    """
    """ counts the number of appearances of each token. returns Map<token, count> """

    frequencies = {}
    for token in my_tokens:

        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1

    return frequencies



def find_intersections(text_1_frequencies, text_2_frequencies):
    counter = 0
    for key in text_1_frequencies.keys():
        if key in text_2_frequencies:
            counter += 1

    return counter


def remove_stop_words(freq):
    # nltk.download('stopwords') #downloaded once only so i comment out
    # Load stop words
    stop_words = stopwords.words('english')
    stop_words = sorted(stop_words)
    # [word for word in tokenized_words if word not in stop_words]

    new_words = []
    for k, v in freq.items():
        if k not in stop_words:
            new_words.append((k, v))

    return new_words


# O(n log n) where n is the number of unique roman character strings in the text file
def get_50_most_common_words(token_freq):
    # sort tokens by decreasing frequencies,
    # Note '-kv[1]' is negative in order to sort by decreasing value
    # inspiration from https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
    tokens = remove_stop_words(token_freq)

    sorted_frequencies = sorted(tokens, key=lambda kv: (-kv[1], kv[0]))
    fifty = sorted_frequencies[0:50]
    return fifty


def sort_alpha(subdomains):
    sorted_subdomains_list = sorted(subdomains, key=lambda kv: (kv[0], kv[1]))

    new_sorted_subdomains_list = []

    # sorted_subdomains_dict = {}
    for domain_name in sorted_subdomains_list:
        # sorted_subdomains_dict[domain_name] = subdomains[domain_name]
        new_sorted_subdomains_list.append((domain_name, subdomains[domain_name]))

    return new_sorted_subdomains_list
