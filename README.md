src/crawler/spiders/get_links.py - Scrapes web pages for text related to U.S. Law Codes

src/data_parser/haiku_parser.py - Loads into memory a list of haiku that are stored in a text file
src/data_parser/tokenizer.py - Formats text, removing unimportant text such as dates, converting to lowercase, etc.

src/markov_chain/compiled_1_order_model.json - First order Markov chain model
src/markov_chain/compiled_2_order_model.json - Second order Markov chain model
src/markov_chain/compiled_3_order_model.json - Third order Markov chain model
src/markov_chain/build_model.py - Loads in text and builds the Markov chain model
src/markov_chain/gen_haiku.py - Generates a haiku from the markov chain model and prints the haiku.
src/markov_chain/markov_chain.py - Keeps a running total of syllables in each line of the haiku for the markov haiku generator
src/markov_chain/sylco.py - Heuristic approach to counting syllables in a word (We did not write this code)

src/ml/decision_trees.py - Creates a decision tree to model the structure of a haiku based off its parts of speech
src/ml/heuristics.py - Defines a heuristic for the RNN to ensure that sentences end with a part of speech common to real haiku
src/ml/Launcher.py - Tests the logistic regressor, decision tree, and heuristic.
src/ml/logistic_regressor.py - Creates a Logistic regressor to model the structure of a haiku based off its parts of speech

src/res/constitution.txt - Plain text of the entire us constitution.
src/res/haiku.csv - Filtered list of haiku (only 5-7-5).
src/res/haiku_full.csv - List of entire haiku dataset (not strictly following 5-7-5).
src/res/parsed_data.txt - List of U.S. law codes
src/res/raw_data.json - List of U.S. law codes in raw html.
src/res/useful_information - Statistics about our dataset (number of words, lines, etc.)

src/rnn/checkpoint_model/model-oliver-twist.pth - Sample RNN model
src/rnn/checkpoint_model/model-us-law-code.pth - RNN model on US law codes
src/rnn/context.py - Utility file to manage system path
src/rnn/generate_text.py - Generates all three lines of a haiku using the RNN.
src/rnn/get_heuristic_model.p - List of part of speech tags used for the RNN heuristic 
src/rnn/oliver.txt - Sample text file for RNN training
src/rnn/parsed_data.txt - List of US law codes
src/rnn_gen.py - Creates a RNN with the US law code text
src/sylco.py - Heuristic approach to counting syllables in a word (We did not write this code)
