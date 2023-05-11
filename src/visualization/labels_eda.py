import collections
import itertools
from collections import Counter

import pandas as pd

from src.models.processhelpers import holdout_directory, load_labels

labels_array = load_labels(holdout_directory)
# count all characters in labels_array
char_counter: Counter = collections.Counter(
    itertools.chain.from_iterable(labels_array),
)
counter_df = pd.DataFrame.from_dict(char_counter, orient='index')
# counter_df = pd.DataFrame.from_dict(counter_dict, orient='index',
#                                     columns=counter_dict.keys)
# print(counter_df)
df_display = counter_df.iloc[:, 0].sort_values(axis=0, ascending=False)
print(df_display)
# sns.displot(df_display)
# plt.show()
# ! dataset lacks letters of Q, W, X
