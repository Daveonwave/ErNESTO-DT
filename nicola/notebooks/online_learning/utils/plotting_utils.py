import seaborn as  sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def pairwise_scatter(df):
    sns.pairplot(df)
    plt.show()

def threed_scatter(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['r0'], df['rc'], df['c'], color='yellow')
    ax.set_xlabel('r0')
    ax.set_ylabel('rc')
    ax.set_zlabel('c')
    plt.show()

def histograms(df):
    df.hist(bins=15, figsize=(15, 6), layout=(1, 3))
    plt.show()

def correlation_heatmap(df):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def marginal_distributions(df):
    for column in df.columns:
        sns.displot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

def samples_and_outliers(samples, outliers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='blue', label='Samples', alpha=0.5)

    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='red', label='Outliers', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def qq_plots(data):
    for column in data.columns:
        stats.probplot(data[column], dist="norm", plot=plt)
        plt.title(f'Q-Q plot for {column}')
        plt.show()