import tqdm
from functools import *
from multiprocess import Process

import math
import numpy as np

import seaborn as sns
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from utils import *
from compute_matrices import *

""" Given the transition matrix of the form CC**{rounds-1}, return the TV
    distance of the random walk from the stationary distribution.

    Keyword arguments:
    matrix -- transition matrix equal to CC**{rounds-1}
    CF -- CF matrix
    FC -- FC matrix
    rounds -- number of rounds that we are interested in
        note that rounds is actually one round lower than the number of
        repetitions of the block cipher. This is because
        (FC@CF)^r = FC@(CF@FC)^{r-1}@CF = FC@CC^{rounds-1}@CF
"""
def get_distance(matrix, CF, FC, rounds):
    # scale of matrix
    matrix_scale = (CC_SCALE)**rounds

    ### compute stationary distribution ###
    # initialize distribution vector
    stationary_distr = np.ones(F, dtype=object)

    # probability of each layout is related to the number of plaintexts
    # and is equal to 255^{# of non-identical entries}
    for full_index in range(F):
        layout_weight = bin(full_index+1).count('1')
        stationary_distr[full_index] = 255**layout_weight
    # scale of stationary_distribution
    stationary_distr_scale = 2**128 - 1

    distribution = FC[0, :]@matrix@CF
    
    # verify that both distributions sum to 1
    if (sum(distribution)*stationary_distr_scale
        != sum(stationary_distr)*matrix_scale):
        return 1

    # compute TV distance
    total_difference = 0
    for i in range(F):
        total_difference += abs(distribution[i]*stationary_distr_scale
                            - stationary_distr[i]*matrix_scale)

    # return log_2 of the distance
    return (math.log2(total_difference)
            -math.log2(matrix_scale*stationary_distr_scale)
            -math.log2(2))

""" Table 2: Statistical distance from pairwise independence of the r-round SPN
        with AES mixing and random S-boxes, given two inputs that differ in exactly
        one coordinate. This corresponds to starting from a layout I with Hamming
        weight 1.
"""

if __name__ == "__main__":

    """
    (README): change precomputed to True to avoid computing the matrices
        CC, CF, FC from scratch
    The matrices should be under the files/ folder
    """
    precomputed = True

    CF = None
    FC = None
    CC = None

    if precomputed:
        print('Found transition matrices in files/, loading...')
        CF = np.load('files/CF.npy', allow_pickle=True)
        FC = np.load('files/FC.npy', allow_pickle=True)
        CC = np.load('files/CC.npy', allow_pickle=True)
    else:
        print('No transition matrices found, computing from scratch...')
        CF = compute_CF_matrix()
        FC = compute_FC_matrix()
        CC = compute_CC_matrix()

    max_power = 10

    # compute powers of CC
    print('Computing the powers of the transition matrix...', end="")
    CC_powers = [CC]
    for i in range(2, max_power+1):
        CC_powers.append(CC_powers[-1]@CC)
    print('DONE')

    """ Table 2. Plot distances from layout with Hamming weight 1. """

    x_list = []
    y_list = []
    table_list = []
    for i in range(max_power):
        rounds = i+2 # CC corresponds to 2 rounds, CC^2 to 3 etc
        distance = get_distance(CC_powers[i], CF, FC, rounds)
        
        x_list.append(rounds)
        y_list.append(distance)
        table_list.append([rounds, distance])

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.xlabel('Number of rounds')
    plt.ylabel('log_2(distance from pairwise)')

    sns.lineplot(x=x_list, y=y_list, marker='o')
    sns.lineplot(x=x_list, y=[-128 for x in x_list], label='desired security')

    # save figure
    plt.savefig('pairwise_dist_AES.png')

    # print distances
    print(tabulate(table_list, headers=['# Rounds', 'Distance from Pairwise']))
