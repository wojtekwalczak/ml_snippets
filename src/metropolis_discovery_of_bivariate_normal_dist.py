#!/usr/bin/env python3

"""Walks through the bivariate normal posterior distribution in a Metropolis-like manner.

Prints the heatmap of the steps taken during the walk, and histogram of visited positions
after the walk."""

from random import uniform
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm

STEPS = 100000
DRAW_EVERY = 10000
DRAW_ALL_STEPS_FOR_FIRST = 50
SHAPE = 100

# posterior distribution parameters
mean = [50, 50]
covariance = [[20., 0.], [0., 20.]]

# for a vector of quantiles return their densities
get_probability = lambda quantiles: multivariate_normal.pdf(quantiles, mean=mean, cov=covariance)

# get the step size in xy directions
get_step = lambda x, y: np.round(norm.rvs(loc=[x, y], scale=3, size=2)).astype(np.int32)

# heatmap
points = np.zeros(shape=(SHAPE, SHAPE), dtype=np.int32)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, SHAPE)
ax.set_ylim(0, SHAPE)

def plot_dot(idx, cur_points, misses):
    ax.set_title('cur_step={}. misses: {}'.format(idx, misses))
    ax.imshow(cur_points)
    fig.canvas.draw()
    fig.canvas.flush_events()

# initial position
init_position = lambda: np.random.randint(0, SHAPE, size=2)
cur_pos = init_position()

# at the end, a histogram of visited positions will be displayed
histogram_data = []

low_probability_count = 0
out_of_map_count = 0
missed_threshold = 0

for i in range(1, STEPS):
    cur_prob = get_probability(cur_pos)
    next_pos = get_step(*cur_pos)
    next_prob = get_probability(next_pos)

    # don't go where the density is very low
    if next_prob < 1e-15:
        # we pick an entirely new position at this point
        cur_pos = init_position()
        low_probability_count += 1
    # don't go out of map
    elif (next_pos >= SHAPE).any():
        out_of_map_count += 1
    # next position has higher density - go there 100%
    elif next_prob > cur_prob:
        cur_pos = next_pos
        points[tuple(cur_pos)] += 1
        histogram_data.extend(cur_pos)
    else:
        # probability of accepting the proposed location
        threshold = next_prob / cur_prob
        # this should not happen due to first 'if'
        if np.isnan(threshold) or np.isinf(threshold):
            low_probability_count += 1
        # go to the new location probabilistically
        elif threshold > uniform(0, 1):
            cur_pos = next_pos
            points[tuple(cur_pos)] += 1
            histogram_data.extend(cur_pos)
        else:
            missed_threshold += 1
    if i % DRAW_EVERY == 0 or i < DRAW_ALL_STEPS_FOR_FIRST:
        print('step: {}. missed_threshold: {}. low_probability_count: {}. out_of_map_count: {}'
              .format(i, missed_threshold, low_probability_count, out_of_map_count))
        plot_dot(i, points, missed_threshold + low_probability_count + out_of_map_count)

# clear the heatmap and plot the histogram
plt.clf()
plt.ioff()
plt.hist(histogram_data, bins=len(np.unique((histogram_data))))
plt.show()