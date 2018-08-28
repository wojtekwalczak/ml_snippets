#!/usr/bin/env python3

# Simple k-means implementation with animated plot.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification

sns.set(style="white", color_codes=True)
customPalette = pd.Series(['#630C3A', '#39C8C6', '#D3500C'])
labelPalette = pd.Series(['#EEEE99', '#DDDD99','#FFB139'])

N_CLASSES=3
N_FEATURES=2

X, y = make_classification(n_samples=500,
                           n_features=N_FEATURES,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=N_CLASSES,
                           n_clusters_per_class=1,
                           class_sep=2.0,
                           hypercube=True,
                           shuffle=False)

X1 = X[:, 0]
X2 = X[:, 1]

df = pd.DataFrame({'x1': X1, 'x2': X2, 'true_label': y, 'cur_label': y})

centroids = np.random.randn(N_CLASSES, N_FEATURES)
cent_labels = np.arange(0, N_CLASSES, 1)
cen_df = pd.DataFrame({'x1': centroids[:, 0], 'x2': centroids[:, 1], 'cur_label': cent_labels})


for i in range(20):
    print('iteration', i)
    plt.cla()
    plt.clf()
    plt.scatter(x=df['x1'],     y=df['x2'],     color=customPalette[df['cur_label']])
    plt.scatter(x=cen_df['x1'], y=cen_df['x2'], color=labelPalette[cen_df['cur_label']])
    plt.pause(0.1)

    def pick_closest(row):
        # K-means happens here. NOTE: `pick_closest()` is nested since it relies on updating `cen_df`.
        return np.argmin(np.sum(np.sqrt(np.square(cen_df[['x1', 'x2']] - row)), axis=1))

    df['cur_label'] = df[['x1', 'x2']].apply(pick_closest, axis=1)
    # recompute centroids
    cen_df = df.groupby(['cur_label'])['x1', 'x2'].mean().reset_index()

# F-score won't work, but V-measure will, because it is independent of the absolute values of the labels.
from sklearn.metrics import v_measure_score
print('V-measure:', v_measure_score(labels_true=df['true_label'], labels_pred=df['cur_label']))
