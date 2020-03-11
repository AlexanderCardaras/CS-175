from data_parser import haiku_parser as hp
from ml import logistic_regressor as lr
from ml import decision_trees as dt
from ml import heuristics as he
import numpy as np

raw_data_path = "res/raw_data.json"         # US code html
parsed_data_path = "res/parsed_data.txt"    # US code parsed
constitution_path = "res/constitution.txt"  # Constitution parsed
haiku_path = "res/haiku.csv"                # Constitution parsed
import pickle

""" N-gram demo """
# data = tokenizer.file_to_list(parsed_data_path)
# n_grams = tokenizer.create_n_grams(data, max_n_gram=1)
# print(n_grams)

""" Logistic regression demo """
# poems, syllables = hp.get_poems(haiku_path, sample_size=1)
# poems, syllables = hp.filter_poems(poems, syllables)
# model = lr.train(poems)
#
# print("\n\nSamples:")
# print("poem:", poems[0])
# print("input:", poems[0][0])
# print("prediction:", lr.classify(poems[0][0], model))
# print("input:", poems[0][1])
# print("prediction:", lr.classify(poems[0][1], model))
# print("input:", poems[0][2])
# print("prediction:", lr.classify(poems[0][2], model))
#
# print("\n")
# print("poem:", poems[1])
# print("input:", poems[1][0])
# print("prediction:", lr.classify(poems[1][0], model))
# print("input:", poems[1][1])
# print("prediction:", lr.classify(poems[1][1], model))
# print("input:", poems[1][2])
# print("prediction:", lr.classify(poems[1][2], model))
#
# print("\n")
# print("poem:", poems[2])
# print("input:", poems[2][0])
# print("prediction:", lr.classify(poems[2][0], model))
# print("input:", poems[2][1])
# print("prediction:", lr.classify(poems[2][1], model))
# print("input:", poems[2][2])
# print("prediction:", lr.classify(poems[2][2], model))


""" Decision Tree demo """
# poems, syllables = hp.get_poems(haiku_path, sample_size=1)
# poems, syllables = hp.filter_poems(poems, syllables)
# model = dt.train(poems)
#
# print("\n\nSamples:")
# print("poem:", poems[0])
# print("input:", poems[0][0])
# print("prediction:", lr.classify(poems[0][0], model))
# print("input:", poems[0][1])
# print("prediction:", lr.classify(poems[0][1], model))
# print("input:", poems[0][2])
# print("prediction:", lr.classify(poems[0][2], model))
#
# print("\n")
# print("poem:", poems[1])
# print("input:", poems[1][0])
# print("prediction:", lr.classify(poems[1][0], model))
# print("input:", poems[1][1])
# print("prediction:", lr.classify(poems[1][1], model))
# print("input:", poems[1][2])
# print("prediction:", lr.classify(poems[1][2], model))
#
# print("\n")
# print("poem:", poems[2])
# print("input:", poems[2][0])
# print("prediction:", lr.classify(poems[2][0], model))
# print("input:", poems[2][1])
# print("prediction:", lr.classify(poems[2][1], model))
# print("input:", poems[2][2])
# print("prediction:", lr.classify(poems[2][2], model))


""" Heuristic demo """
poems, syllables = hp.get_poems(haiku_path, sample_size=1)
poems, syllables = hp.filter_poems(poems, syllables)
model = he.get_heuristic(poems)
# with open("get_heuristic_model.p", "wb") as f:
#     pickle.dump(model, f)

print("input:", poems[0][0])
print("prediction:", he.classify(poems[0][0], model))

print("input:", poems[0][1])
print("prediction:", he.classify(poems[0][1], model))

print("input:", poems[0][2])
print("prediction:", he.classify(poems[0][2], model))

print("input:", poems[1][0])
print("prediction:", he.classify(poems[1][0], model))

print("input:", poems[1][1])
print("prediction:", he.classify(poems[1][1], model))

print("input:", poems[1][2])
print("prediction:", he.classify(poems[1][2], model))

test = "not a thing that he"
print("input:", test)
print("prediction:", he.classify(test, model))


