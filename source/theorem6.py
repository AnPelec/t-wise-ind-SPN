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

""" Verifies that the distance from pairwise is at most 2**(-limit)
    if we start from layout with index idx.

    Keyword arguments:
    idx -- the index of the (full) layout that we are starting from
    rounds -- number of rounds that we are interested in
        note that rounds is actually one round lower than the number of
        repetitions of the block cipher. This is because
        (FC@CF)^r = FC@(CF@FC)^{r-1}@CF = FC@CC^{rounds-1}@CF
    matrix -- transition matrix equal to CC**{rounds-1}
    limit -- the distance we want to verify

    Output:
     * Function outputs the indices of the layouts that are not close enough
       in files/violate_{rounds}_{limit}.txt
"""
def verify_close(idx, rounds, matrix, limit):

    # load matrices since we are running the code in parallel
    CF = np.load('files/CF.npy', allow_pickle=True)
    FC = np.load('files/FC.npy', allow_pickle=True)

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

    distribution = FC[idx, :]@matrix@CF

    # verify that both distributions sum to 1
    if (sum(distribution)*stationary_distr_scale
        != sum(stationary_distr)*matrix_scale):
        with open(f'files/violate_{rounds}_{limit}.txt', 'a') as f:
            f.write(f'incorrect normalization with idx {idx}\n')
        return

    # compute TV distance
    total_difference = 0
    for i in range(F):
        total_difference += abs(distribution[i]*stationary_distr_scale
                            - stationary_distr[i]*matrix_scale)

    # verify distance bound
    if (math.log2(total_difference)
        -math.log2(matrix_scale*stationary_distr_scale)
        -math.log2(2)) > -limit:
        with open(f'files/violate_{rounds}_{limit}.txt', 'a') as f:
            f.write(f'idx {idx} is at distance more than 2^{-limit}\n')


""" Theorem 6: The 7-round AES* is 2^{-128.0}-close to pairwise independent.
"""
if __name__ == "__main__":

    """
    (README): change precomputed to True to avoid computing the matrices
        CC, CF, FC from scratch
    The matrices should be under the files/ folder
    """
    precomputed = False

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
    print('Computing the powers of the transition matrix...')
    CC_powers = [CC]
    for i in range(2, max_power+1):
        CC_powers.append(CC_powers[-1]@CC)
    print('DONE')

    """ Theorem 6. Verify 7 rounds are 2^{-128} close. """

    parallel = 10

    for i in tqdm.tqdm(range(0, F, parallel)):
        # run 10 layouts in parallel and verify that 7 rounds
        # are at least 2^{-128}-close to pairwise
        processes = []
        
        for r in range(parallel):
            if i + r < F:
                processes.append(Process(target=verify_close,
                                        args=(i+r, 7, CC_powers[5], 128.0)))
                processes[-1].start()
        
        processes[-1].join()
    
    with open(f'files/violate_{7}_{128.0}.txt', 'a') as f:
        f.write(f'DONE\n')
