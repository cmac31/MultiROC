#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

This code implements the MultiROC method by E Shult (Schult, E. K. (1995). Multtivariate Receiver-Operating Screening as an Example Characteristic Curve Analysis : Prostate Cancer screening as an example. Clinical Chemistry, 41(8), 1248–1255.).
Input should be a 3 dataframe with binary categorisation labels as the first column, and marker values or probabilities from logistic regression as the second and third column.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MultiROC:


    def __init__(self):
        self.cutoff_1 = None,
        self.cutoff_2 = None
        self.AUC_ = None
        self.true_pos = []
        self.false_pos = []
        self.true_pos_2 = []
        self.false_pos_2 = []

    def fit(self, df):
        """
        Fit model coefficients.

        Arguments:
        X: pandas dataframe with variable of interet as second column
        y: 1D numpy array
        """

        # check if X is 1D or 2D array
        X = df.iloc[:,1]
        y = df.iloc[:,0]
        #if len(X.shape) == 1:
        #    X = X.reshape(-1,1)

        negative_count = len(y[y == 0])
        positive_count = len(y[y == 1])
        tpr = 0
        fpr = 0
        tpr_score = []
        fpr_score = []

        for row in y:
            if row == 1:
                tpr += 1.0/positive_count
                tpr_score.append(tpr)
                fpr_score.append(fpr)
            elif row == 0:
                fpr += 1.0/negative_count
                tpr_score.append(tpr)
                fpr_score.append(fpr)

        cutoff = X[np.argmax(np.array(fpr_score) - np.array(tpr_score))]

        # set attributes
        self.true_pos = tpr_score
        self.false_pos = fpr_score
        self.cutoff_1 = cutoff

        df = df[df.loc[:,1] > self.cutoff_1]
        W = df.iloc[:,2]
        y = df.iloc[:,0]

        negative_count_2 = len(y[y == 0])
        positive_count_2 = len(y[y == 1])
        tpr_2 = 0
        fpr_2 = 0
        tpr_score_2 = []
        fpr_score_2 = []

        for row in y:
            if row == 1:
                tpr_2 += 1.0/positive_count_2
                tpr_score_2.append(tpr_2)
                fpr_score_2.append(fpr_2)
            elif row == 0:
                fpr_2 += 1.0/negative_count_2
                tpr_score_2.append(tpr_2)
                fpr_score_2.append(fpr_2)

        cutoff_2 = W[np.argmax(np.array(fpr_score_2) - np.array(tpr_score_2))]

        # set attributes
        self.true_pos_2 = tpr_score_2
        self.false_pos_2 = fpr_score_2
        self.cutoff_2 = cutoff_2

    def generate_curve(self):
        plt.plot(self.false_pos, self.true_pos, label = 'marker 1')
        plt.plot(self.false_pos_2, self.true_pos_2, label = 'marker 1 + 2')
        plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), '--', color = 'black')
        plt.xlabel('1 - sensitvity')
        plt.ylabel('specificity')
        plt.show()


X = pd.DataFrame([np.random.random_integers(0,1,100), np.random.random_integers(0, 50, 100),
                  np.random.random_integers(60,70,100)]).T

test = MultiROC()
test.fit(X)

print(test.cutoff_1,
      test.cutoff_2)

test.generate_curve()
