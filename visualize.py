# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:36:40 2022

@author: ugo.pelissier
"""

from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
)
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

# Generate the class/group data
n_points = 100
X = rng.randn(100, 10)

p_true = 0.25
percentiles_classes = [p_true, 1-p_true]
y = np.hstack([[ii] * int(100 * perc)
              for ii, perc in enumerate(percentiles_classes)])

# Generate uneven groups
n_groups = 4
group_prior = rng.dirichlet([2] * n_groups)
groups = np.repeat(np.arange(n_groups), rng.multinomial(int(100*p_true), group_prior))
groups = np.concatenate((groups, [n_groups for i in range(100-int(100*p_true))]), axis=0)


def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["Data\ngroup", "Data\nclass"],
        xlabel="Sample index",
    )


visualize_groups(y, groups, "no groups")


def plot_cv_indices(X, y, group, n_groups, p_true, ax, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    offset = int(100*p_true)
    for i in range(n_groups):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))      
        count = 0
        for j in range(len(indices)):  
            if groups[j]==i:
                indices[j] = 1
                count += 1
            else: indices[j] = 0
        
        n = 3*count
        discard = []
        for j in range(offset,offset+n):
            indices[j] = 1
            discard.append(j)
        train_false = []
        for j in range(int(100*p_true),len(X)):
            if j not in discard:
                train_false.append(j)
        rd = random.sample(train_false, count)
        for r in rd:
            indices[r] = 2
        offset += n
        
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [i + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [i + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [i + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_groups)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_groups + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_groups + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("Stratified Group KFold", fontsize=15)
    return ax

fig, ax = plt.subplots()
plot_cv_indices(X, y, groups, n_groups, p_true, ax)
ax.legend(
    [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02)), Patch(color=cmap_cv(2.0))],
    ["Testing set", "Training set", "Random Downsampling"],
    loc=(1.02, 0.8),
)