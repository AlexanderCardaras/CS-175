import markovify

"""
Build a markov chain model from raw text files and exports the compiled model
as a json file
"""

if __name__ == "__main__":
  # Gets raw text as strings
  with open('res/constitution.txt') as f:
    constitution_text = f.read()
  with open('res/parsed_data.txt') as f:
    us_law_text = f.read()

  # Builds the models
  constitution_model = markovify.Text(constitution_text)
  us_law_model = markovify.Text(us_law_text)

  # Combines the models
  text_model = markovify.combine([constitution_model, us_law_model]).compile()

  # Exports the compiled model
  with open('markov_chain/compiled_model.json', 'w') as f:
    f.write(text_model.to_json())
