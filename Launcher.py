from data_parser import tokenizer
from ml import logistic_regressor
from data_parser import haiku

raw_data_path = "res/raw_data.json"         # US code html
parsed_data_path = "res/parsed_data.txt"    # US code parsed
constitution_path = "res/constitution.txt"  # Constitution parsed
haiku_path = "res/haiku.csv"                # Input haiku dataset


# laws = tokenizer.strip_json(raw_data_path)
# tokenizer.list_to_file(laws, parsed_data_path)

# max_n = 7
#
# data = tokenizer.file_to_list(parsed_data_path)
# n_grams = tokenizer.create_n_grams(data, max_n_gram=max_n)
#
# data = tokenizer.file_to_list(constitution_path)
# n_grams = tokenizer.create_n_grams(data, n_grams=n_grams, max_n_gram=max_n)
#
# for n in n_grams.keys():
#     print("n={} {}".format(n, len(n_grams[n])))


lines = haiku.file_to_list(haiku_path)
poems, syllables = haiku.lines_to_poems(lines)
logistic_regressor.train(poems)
