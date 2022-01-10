# Importing libraries ...
import numpy as np
import matplotlib.pyplot as plt


# ==== Plot loss and accuracy ====
def plotLossAndAccuracy(dataframe, fname, cut_epochs):
    """
    Plot loss and accuracy and save extract as svg format
    :param dataframe: experiment history dataframe
    :param fname: picture path
    :param cut_epochs: the epochs which finetuning occurs (0 if we do only feature extractor)
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(30, 12))
    num_epochs = len(dataframe)

    # Plot Loss result
    axs[0].plot(np.arange(num_epochs), dataframe['Train_loss'], label='Train loss', color='b')
    axs[0].plot(np.arange(num_epochs), dataframe['Val_loss'], label='Val loss', color='r')
    axs[0].set_xlabel("Epochs", fontsize=15)
    axs[0].set_ylabel("Loss", fontsize=15)
    axs[0].set_title("Loss Plot", fontsize=18)
    if cut_epochs > 0:
        axs[0].vlines(cut_epochs, 0, 1.6, linestyle='--', color='g')
    axs[0].legend(fontsize=14)
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[0].yaxis.set_tick_params(labelsize=14)

    # Plot Accuracy result
    axs[1].plot(np.arange(num_epochs), dataframe['Train_acc'], label='Train Accuracy', color='b')
    axs[1].plot(np.arange(num_epochs), dataframe['Val_acc'], label='Val Accuracy', color='r')
    axs[1].set_xlabel("Epochs", fontsize=15)
    axs[1].set_ylabel("Accuracy", fontsize=15)

    # Find the best model
    max_id = dataframe['Val_acc'].argmax()

    axs[1].set_title(
        f"Best Validation Accuracy: {dataframe['Val_acc'][max_id]:.2f}%\nBest Train Accuracy: {dataframe['Train_acc'][max_id]:.2f}%",
        fontsize=18)
    axs[1].plot(max_id, dataframe['Val_acc'][max_id], 'ro', markersize=12)
    axs[1].text(max_id, dataframe['Val_acc'][max_id] + 1, f'Best Model\nEpochs {max_id}', va='bottom', ha='center',
                fontsize=14)
    if cut_epochs > 0:
        axs[1].vlines(cut_epochs, 40, 100, linestyle='--', color='g')
    axs[1].legend(fontsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[1].yaxis.set_tick_params(labelsize=14)

    # Save figure
    fig.savefig(fname, format='svg', dpi=1200)
