import markovify

"""
Build a markov chain model from raw text files and exports the compiled model
as a json file for each order of markov chain (1 to 4)
"""

if __name__ == "__main__":
  # Gets raw text as strings
  with open('res/constitution.txt') as f:
    constitution_text = f.read()
  with open('res/parsed_data.txt') as f:
    us_law_text = f.read()

  # Order of markov chain
  for i in range(1,4):

    # Builds the models
    constitution_model = markovify.Text(constitution_text, state_size=i)
    us_law_model = markovify.Text(us_law_text, state_size=i)

    # Combines the models
    text_model = markovify.combine([constitution_model, us_law_model]).compile()

    # Exports the compiled model
    with open('markov_chain/compiled_' + str(i) + '_order_model.json', 'w') as f:
      f.write(text_model.to_json())
