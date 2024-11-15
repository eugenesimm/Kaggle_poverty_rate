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