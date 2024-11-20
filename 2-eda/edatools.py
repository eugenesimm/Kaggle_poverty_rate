import seaborn as sns
import matplotlib.pyplot as plt

def correlational_matrix_heatmap(data, fig_size=(6,6), cols_to_exclude=[]):
    data_variates = data.drop(columns=cols_to_exclude)

    cor_matrix = data_variates.corr()
    
    # Plot heatmap
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)

    # Move x-axis labels to the top
    ax.xaxis.tick_top()  # Moves x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Sets the label position to the top

    plt.show()


def correlation_bar_graph(data, y_col='subjectivePoverty_rating', fig_size=(20, 8), x_cols_to_exclude=[]):


    # Assuming 'correlations' is a Series containing correlations with 'rating'
    # Example: correlations = merged_df.corr()['rating'].sort_values(ascending=False)
    cols = [col for col in data.columns if col not in x_cols_to_exclude]
    correlations = data[cols].corr()[y_col].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != y_col]

    # Plot a bar graph
    plt.figure(figsize=fig_size)
    correlations.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Absolute Value of Correlations with {y_col}', fontsize=16)
    plt.xlabel('Explanatory variables', fontsize=14)
    plt.ylabel(f'Correlation with {y_col}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

def density_plot(data):

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, density=True, edgecolor='k', alpha=0.7)
    plt.title('Density Histogram Data')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()