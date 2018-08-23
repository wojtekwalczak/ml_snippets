#!/usr/bin/env python3

"""Draws `W = tanh(W_hat) * sigmoid(M_hat)`, where `*` is element-wise multiplication.

The formula is taken from the following paper: https://arxiv.org/pdf/1808.00508.pdf

The question is: why on earth the surface resulting from element-wise multiplication
of tanh and sigmoid makes sense when one would like to learn `W` to be (approximately)
one of {-1, 0, 1}.

The answer is: this kind of surface has stationary points near -1 and 1 (like `tanh`),
but also kind of mezzanine floor close to 0.
"""

import numpy as np
import matplotlib.pyplot as plt

# This import is necessary for 3d projection
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1/(1+np.exp(-x))

def count_W(W_hat, M_hat):
    """See page 3 of: https://arxiv.org/pdf/1808.00508.pdf"""
    return np.multiply(np.tanh(W_hat), sigmoid(M_hat))

xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(xs, ys)

W = count_W(X.ravel(), Y.ravel())
W = W.reshape(X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X=X, Y=Y, Z=W)
plt.show()