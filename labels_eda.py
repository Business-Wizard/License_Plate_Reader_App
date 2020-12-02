from src.models.train_model import load_labels, holdout_directory
from src.models import segmentation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

labels_array = load_labels(holdout_directory)
counter_dict = dict()
for idx, label in enumerate(labels_array):
    for char in label:
        if counter_dict.get(char, False):
            counter_dict[char] += 1
        else:
            counter_dict[char] = 1
counter_df = pd.DataFrame.from_dict(counter_dict, orient='index')
# counter_df = pd.DataFrame.from_dict(counter_dict, orient='index', columns=counter_dict.keys)
# print(counter_df)
df_display = counter_df.iloc[:,0].sort_values(axis=0, ascending=False)
sns.displot(df_display)
plt.show()
# ! dataset lacks letters of Q, W, X