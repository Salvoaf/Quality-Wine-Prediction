import matplotlib
import matplotlib as plt
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns


def correlationMatrix(df):
    """
    This function to print the correlation matrix.

    Parameters
    ----------
    df: DataFrame
        the competition DataFrame
    """
    # -----------------------------correlation matrix------------------------------------------------#

    corr = df.corr()
    matplotlib.pyplot.subplots(figsize=(15, 10))
    matplotlib.pyplot.title(f'Matrice di correlazione')
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
                cmap=sns.diverging_palette(220, 20, as_cmap=True))
    plt.show()


def BoxPlot(df):
    """
    This function prints the boxplot.
    Boxplot is a method for graphically demonstrating the locality,
    spread and skewness groups of numerical data through their quartiles.

    Parameters
    ----------
    df: DataFrame
        the competition DataFrame
    """
    # -----------------------------BOXPLOT OUTLIER------------------------------------------------#
    # stampo tutti i plotbox
    pl.figure(figsize=(8, 7))
    l = df.columns.values
    number_of_columns = 12
    number_of_rows_for_figure = 1
    plt.figure(figsize=(number_of_columns, 3 * 1))
    for i in range(0, len(l) - 1):
        a1 = plt.subplot(1, len(l) - 1, i + 1)
        sns.set_style('whitegrid')
        sns.boxplot(y=df[l[i]], color='green', orient='v')

    plt.tight_layout()
    plt.show()


