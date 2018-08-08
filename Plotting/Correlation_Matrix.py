import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# todo Adding comment and documentation
def correlation_matrix(df, cols, font_scale=1.5):
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=font_scale)
    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 15},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.show()