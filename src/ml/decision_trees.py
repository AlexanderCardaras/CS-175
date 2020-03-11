from sklearn.model_selection import train_test_split
from sklearn import tree
from nltk.data import load
import numpy as np
from src.data_parser import haiku_parser as hp
from sklearn.metrics import roc_curve, auc


tags = list(load('help/tagsets/upenn_tagset.pickle').keys())


def pos_to_int(pos):
    """
    Converts a words part of speech into an integer by referencing the index of pos in the tags array
    :param pos: part of speech nltk naming convention
    :return: index representation of pos + 1
    """

    return tags.index(pos)+1


def line_to_int(line, cols=7):
    """
    Converts a line of parts of speech into a list of ints
    :param line: string of words
    :param cols: what is the minimum size of the output list (how much buffer do I need)
    :return: list(length 7) of ints filled with pos int values and 0 for buffer values
            Returns None when nan is given as input
    """

    line_num = []

    # Some lines are read in as nan. Dont know why, but catch the AttributeError when it happens
    try:

        # Convert line to list of parts of speech
        pos = hp.get_parts_of_speech(line)

        # Convert parts of speech to ints
        for p in pos:
            line_num.append(pos_to_int(p))

        # Pad lists 0 to make a length 7 list
        line_num += [0] * (cols - len(line_num))
        return np.array(line_num)
    except AttributeError:
        return None


def poem_to_int(poem, cols=7):
    """
    :param poem: list of strings
    :param cols: what is the minimum size of the output list (how much buffer do I need)
    :return: A list containing 3 sublists, each representing a list(length 7) of ints filled with pos int values
                and 0 for buffer values
    """

    parts_of_speech = []

    # Loop through each line in the poem
    for line in poem:

        # Convert the line to list of ints
        line_num = line_to_int(line, cols)

        # line_to_int will return None when given nan as input
        if line_num is not None:
            parts_of_speech.append(line_num)

    return parts_of_speech


def get_x(poems, cols=7):
    """
    Converts a list of poems into a matrix representation of their parts of speech
    :param poems: List of poems
    :param cols: number of columns in matrix, default 7
    :return: 7x3 Matrix representation of all poems
    """

    # List of matrices, each representing the parts of speech of a poem
    X = [[], [], []]

    # Loop through all poems
    for poem in poems:

        # Convert the poem into a list of pos indices
        ints = poem_to_int(poem, cols)

        # Some lines return nan, do not accept any poems with any lines missing
        if len(ints) == 3:
            l1 = ints[0]
            l2 = ints[1]
            l3 = ints[2]

            # Fill X
            X[0].append(l1)
            X[1].append(l2)
            X[2].append(l3)

    return np.vstack((X[0], X[1], X[2]))


def get_y(X):
    """
    Constructs a list numbers (0-2) of length equal to X [0,0,0,0, ... 1,1,1,1, ... 2,2,2,2]
    :param X: numpy array containing X training data
    :return: List of corresponding Y training data
    """

    # Calculate how many of each number 0s, 1s and 2s need to be in the list
    size = int(X.shape[0]/3)

    # Concatenate lists
    return np.hstack((np.full(size, 0), np.full(size, 1), np.full(size, 2)))


def train(poems, test_fraction=0.25):
    """
    Train a logistic regressor on the parts of speech of poems
    :param poems: Poems taken from haiku.csv
    :param test_fraction: Test/Train ratio
    :return: Trained classifier
    """

    X = get_x(poems)
    Y = get_y(X)

    # split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)

    # Specify the logistic classifier model with an l2 penalty for regularization and with fit_intercept turned on
    # classifier = linear_model.LogisticRegression(penalty="l2", fit_intercept=True, max_iter=10000)
    classifier = tree.DecisionTreeClassifier(max_depth=10, random_state=42)

    print('\nTraining a model with', X_train.shape[0], 'examples.')

    # Fit the classification model
    classifier.fit(X_train, Y_train)

    # Debugging code
    train_accuracy = classifier.score(X_train, Y_train)
    test_accuracy = classifier.score(X_test, Y_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print('\n\tTraining accuracy:\t', format(100 * train_accuracy, '.2f'))
    print('\n\tTesting accuracy:\t', format(100 * test_accuracy, '.2f'))

    return classifier


def classify(line, classifier):
    """
    Predicts the line number of a line
    :param line: A generated string for a poem
    :param classifier: Model to classify the line
    :return: Predicted line number
    """

    # Convert line to list of ints
    X = line_to_int(line)

    # Reshape, sklearn wants this shape when working with a single sample
    X = X.reshape(1, -1)

    # Predict function returns a list, so take the first and only value from its output
    return classifier.predict(X)[0]

