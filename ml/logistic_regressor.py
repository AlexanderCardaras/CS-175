from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

from data_parser import sentence_splitter
import numpy as np

from data_parser import tokenizer
parts_of_speech =[
"CC",
"CD",
"DT",
"EX",
"FW",
"IN",
"JJ",
"JJR",
"JJS",
"LS",
"MD",
"NN",
"NNS",
"NNP",
"NNPS",
"PDT",
"POS",
"PRP",
"PRP$",
"RB",
"RBR",
"RBS",
"RP",
"TO",
"UH",
"VB",
"VBD",
"VBG",
"VBN",
"VBP",
"VBZ",
"WDT",
"WP",
"WP$",
"WRB",
"SYM"
]


def string_to_number(data):
    numbers = []
    X = data.flatten()
    for line in X:
        try:
            pos = sentence_splitter.get_parts_of_speech(line)
        except AttributeError:
            numbers.append(-1)
            continue

        temp = 0
        for p in pos:
            temp = parts_of_speech.index(p) + temp * 100
            print(temp)
        numbers.append(temp)
    return np.array(numbers)


def train(data, test_fraction=0.5):

    X = string_to_number(data).reshape(-1, 1)

    Y = []
    for i in range(0, len(data)):
        Y.append(0)
        Y.append(1)
        Y.append(2)

    Y = np.array(Y).reshape(-1, 1)

    # split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)

    # Specify the logistic classifier model with an l2 penalty for regularization and with fit_intercept turned on
    classifier = linear_model.LogisticRegression(penalty="l2", fit_intercept=True)

    # Train a logistic regression classifier and evaluate accuracy on the training data
    print('\nTraining a model with', X_train.shape[0], 'examples.....')
    # .... fit the classification model.....
    classifier.fit(X_train, Y_train)
    train_predictions = classifier.predict(X_train[:2, :])
    train_accuracy = classifier.score(X_train, Y_train)
    print('\nTraining:')
    print(' accuracy:', format(100 * train_accuracy, '.2f'))

    # Compute and print accuracy and AUC on the test data
    print('\nTesting: ')
    test_accuracy = classifier.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

    class_probabilities = classifier.predict_proba(X_test)[:, 1]
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities)
    print(' AUC value:', format(100 * test_auc_score, '.2f'))
    return classifier


def classify(data, model):
    return None

