from src.data_parser import haiku_parser as hp


def line_to_pos(line):
    """
    Gets the last part of speech in the line
    :param line: A sentence (string)
    :return: A single part of speech key
    """

    # Get the parts of speech of the line
    pos = hp.get_parts_of_speech(line)

    # return the last element in the pos dictionary
    return list(pos.keys())[-1]


def poem_to_pos(poem):
    """
    Creates a dictionary of ending parts of speech (pos:frequency) of a single poem
    :param poem: list of lines
    :return: Dictionary of parts of speech keys and frequency values
    """

    all_pos = {}

    # Loop through each line of the poem
    for i in range(0, len(poem)):

        # Set the part of speech for current line
        all_pos[i] = line_to_pos(poem[i])

    return all_pos


def poems_to_pos(poems):
    """
    Creates a dictionary of ending parts of speech (pos:frequency) of all poems
    :param poems: list of poems
    :return: Dictionary of parts of speech keys and frequency values
    """

    # List of matrices, each representing the parts of speech of a poem
    X = {}

    # Loop through all poems
    for poem in poems:

        # Convert the poem into a list of end parts of speeches
        pos = poem_to_pos(poem)

        # Add parts of speech to dict
        if pos[0] in X:
            X[pos[0]] += 1
        else:
            X[pos[0]] = 1

        if pos[1] in X:
            X[pos[1]] += 1
        else:
            X[pos[1]] = 1

        if pos[2] in X:
            X[pos[2]] += 1
        else:
            X[pos[2]] = 1

    return X


def get_heuristic(poems, k=5):
    """
    Finds the most common ending parts of speech in poems and returns a set of the top 'k' of them
    :param poems: List of poems
    :param k: number of top parts of speech to keep when filtering
    :return: Top 'k' parts of speech for the ending words of 'poems'
    """

    # Convert poems to parts of speech dict
    pos = poems_to_pos(poems)

    # Sort parts of speech by their frequency
    model = sorted(pos, key=lambda x: pos[x], reverse=True)

    # return the first k elements as a set
    return set(model[0:k])


def classify(line, model):
    """
    Finds if the line has a valid ending, as determined by 'model'
    :param line: A sentence (string)
    :param model: A set of parts of speech
    :return: True if the last part of speech in line is in 'model', False otherwise
    """

    # Get the last pos in the line
    pos = line_to_pos(line)

    # Check if the model contains the pos
    return pos in model
