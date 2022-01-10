# Importing libraries ...
import matplotlib.pyplot as plt

# Catagorical citeria
CATORICAL_SCORE = [[15, 12, 7, 4, 0], [20, 17, 13, 8, 3], [20, 15, 10, 8, 5],
                   [6, 4, 2, 1, 0], [6, 5, 4, 1, 0], [6, 4, 2, 2, 0],
                   [6, 5, 3, 1, 0], [15, 10, 7, 4, 0], [15, 10, 7, 4, 0]]


# ==== Plot distribution of each parameters ====
def summaryParameters(dataframe, fname):
    """
    Plot distribution of all RMR parameters and save as figure
    :param dataframe: totol dataframe (xlsx file)
    :param fname: figure path
    """
    # Create figures
    fig, ax = plt.subplots(3, 3, figsize=(30, 30))
    for catsc, columnname, axs in zip(CATORICAL_SCORE, dataframe.columns[3:12], ax.flatten()):
        a, b, c, d, e = catsc
        data_apply = dataframe[columnname].apply(
            lambda x: 0 if x >= a else (1 if x >= b else (2 if x >= c else (3 if x >= d else 4))))
        bar_data = data_apply.value_counts().sort_index().to_numpy()
        bar_index = data_apply.value_counts().sort_index().index.to_numpy()
        axs.bar(bar_index, bar_data)
        axs.set_title(f"{columnname}", fontsize=18)
        axs.set_xlabel('Categorical Score', fontsize=15)
        axs.set_ylabel('Sampel count', fontsize=15)
        axs.set_xlim([- 0.5, 4.5])
        axs.set_xticks([0, 1, 2, 3, 4])
        axs.set_xticklabels(
            ['>' + str(a), str(b) + '-' + str(a), str(c) + '-' + str(b), str(c) + '-' + str(d), str(e) + '-' + str(d)],
            fontsize=14)
        axs.yaxis.set_tick_params(labelsize=14)
        axs.set_ylim([0, max(data_apply.value_counts()) * 1.20])
    plt.tight_layout()

    # Save figure
    fig.savefig(fname, format='svg', dpi=1200)


# ==== Plot each distribution for each dataframe ====
def plotDistribution(dataframe, title_chart, fname):
    """
    Plot RMR Category distribution for each dataframe
    :param fname: picture path for saving
    :param dataframe: "pandas dataframe"
    :param title_chart: "The name of title for graph"
    :return ax: "The axes of charts"
    """
    ax = dataframe.Small.value_counts().sort_index().plot.bar(figsize = (12, 8))
    ax.set_title(title_chart, fontsize=18)
    ax.set_xlabel("RMR Category", fontsize=15)
    ax.set_ylabel("Sample Count", fontsize=15)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_ylim([0, max(dataframe.Small.value_counts()) * 1.2])
    for x, y in zip(range(7), dataframe.Small.value_counts().sort_index()):
        ax.text(x, y + 20, y, va='top', ha='center', fontsize=14)

    ax.set_xticklabels(["10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
                       , rotation=45)
    ax.get_figure().savefig(fname, format='svg', dpi=1200)
    return ax
