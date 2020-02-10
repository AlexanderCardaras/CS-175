import pandas as pd
from keras.utils import np_utils


def file_to_list(file_path, sample_size=0.01):
    contents = pd.read_csv(file_path)
    return contents.sample(frac=sample_size)


def lines_to_poems(df):

    lines = set([0, 1, 2])

    for i in range(3):
        lines.remove(i)
        df = df[['0', '1', '2',] + ['%s_syllables' % j for j in lines]].join(
            df['%s_syllables' % i].str.split(',', expand=True).
                stack(-1).reset_index(level=1, drop=True).rename('%s_syllables' % i)
        ).drop_duplicates()
        lines.add(i)

    for i in range(3):
        df['%s_in' % i] = (df[str(i)])

        if i == 2:  # If it's the last line
            df['%s_out' % i] = df[str(i)]
        else:
            # If it's the first or second line, add the first character of the next line to the end of this line.
            # This helps with training so that the next RNN has a better chance of getting the first character right.
            df['%s_out' % i] = (df[str(i)] + df[str(i + 1)].str[0])

    inputs = df[['0_in', '1_in', '2_in']].values
    syllables = df[['0_syllables', '1_syllables', '2_syllables']].values

    return inputs, syllables
