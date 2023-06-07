#!/usr/bin/env python3
import numpy as np
from rich import print
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix

class prmetric(object):
    def __init__(self, acc_hit=0, acc_total=1, recall_hit=0, recall_total=1):
        self.acc_hit = acc_hit
        self.acc_total = acc_total
        self.recall_hit = recall_hit
        self.recall_total = recall_total

    def __repr__(self) -> str:
        precision = self.acc_hit / self.acc_total
        recall = self.recall_hit / self.recall_total
        f1 = 2 * precision * recall / (precision + recall)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        return f'precision: {precision}, recall: {recall}, f1: {f1}'


class Printer(object):
    @staticmethod
    def classification_report(y_test, y_pred, label_encode=False):
        """ Print classification report
        """
        if label_encode:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            all_ = list(set(y_test + y_pred))
            le.fit_transform(all_)
            y_test = le.transform(y_test)
            y_pred = le.transform(y_pred)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=1))

        if label_encode:
            for i in range(len(all_)):
                print(f'{i}: {le.inverse_transform([i])[0]}', end=' ')


class Ploter(object):
    @staticmethod
    def confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Args:
            cm(array, shape = [n, n]): a confusion matrix of integer classes
            classes(list): a list of class names
            normalize(bool): whether to normalize the confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
