import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# todo Adding comment and documentation
def scatter_plot_matrix(df, cols, size=2.5):
    sns.pairplot(df[cols], size=size)
    plt.tight_layout()
    plt.show()