import itertools

import numpy as np
from matplotlib import pyplot as plt

#https://dfrieds.com/machine-learning/visual-introduction-classification-logistic-regression-python.html
def plot_cm(cm, class_labels, title, cmap=plt.cm.Blues, savefile=None):
    """Plots a confusion matrix"""
    
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", 
                 fontsize=23, color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True topic', labelpad=12)
    plt.xlabel('Predicted topic', labelpad=12)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    
def plot_cms(cms, class_labels, suptitle, titles, cmap=plt.cm.Blues, figsize=(12,8), savefile=None):
    """Plots a list of confusion matrices"""
    
    fig, axs = plt.subplots(1, len(cms), figsize=figsize)
    for i in range(len(cms)):
        cm = cms[i]
        ax = axs[i]
        labels = class_labels[i]
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(titles[i])
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", 
                     fontsize=16, color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('True topic', labelpad=12)
        ax.set_xlabel('Predicted topic', labelpad=12)
        
    plt.suptitle(suptitle)  
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()