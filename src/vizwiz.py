#!/bin/python
import matplotlib.pyplot as plt   
import scikitplot.metrics as skpm
import numpy as np

class VizWiz:
    '''
        Class VizWiz for visualization things
    '''    
    def plot_class_freq(self, savefig=False, **args):
        """
            Plot counts per class
        """
        fig = plt.figure(num=None, figsize=(15, 20), dpi=100)
        i=1
        for row in range(len(self.y_train)):
            unique, counts = np.unique(self.y_train[row], return_counts=True)
            plt.subplot(len(self.y_train),2,i).bar(unique, counts, label="Train")
            unique, counts = np.unique(self.y_test[row], return_counts=True)
            plt.subplot(len(self.y_train),2,i).bar(unique, counts, label="Test")
            i+=1
            plt.title(f"Class Frequency (n_split={row+1})")
            plt.xlabel("Class")
            plt.ylabel("Frequency")
            plt.legend()
        plt.tight_layout()
        if savefig:
            fig.savefig('class_freqs.png')

    def plot_confusion(self, savefig=False, **args):
        """
            Plot a confusion matrix
        """
        fig = plt.figure(num=None, figsize=(15, 12), dpi=100)
        i=1
        for row in range(len(self.y_train)):
            skpm.plot_confusion_matrix(y_true=self.y_train[row], y_pred=self.y_train_pred[row], title=f"Train Set (n_split={row+1})", ax=fig.add_subplot(len(self.y_train),2,i), **args)
            i+=1
            skpm.plot_confusion_matrix(y_true=self.y_test[row], y_pred=self.y_test_pred[row], title=f"Test Set (n_split={row+1})", ax=fig.add_subplot(len(self.y_train),2,i), **args)
            i+=1
        plt.tight_layout()
        if savefig:
            fig.savefig('confusion_matrix.png')
            
