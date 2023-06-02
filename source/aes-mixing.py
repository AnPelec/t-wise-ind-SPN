import copy
import tqdm
import random
import itertools
from functools import *
from multiprocess import Process

import math
import numpy as np
from mpmath import *
from pyfinite import ffield

import seaborn as sns
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from utils import *

k = 4 # even though AES has 16 blocks, its mixing is full-branch
      # when viewed as a linear mapping from F^4 -> F^4 (i.e. between columns)
K = 16
# a compressed layout stores the Hamming weight of the layout columns
# Hence each Hamming weight can map to 1, 4, or 6 distinct layout columns
# For our matrices to have integer entries, we multiply everything by
# LCM(4, 6)^4 = 20736
COMMON_DENOMINATOR = 20736

# PARALLELIZE MULTIPLICATION

def multiply(start, end, index):
    """ Compute matrix multiplication in parallel.
    We are computing the product of CF@FC between lines [start, end)
    
    Keyword arguments:
    start -- start of the range
    end -- end of the range
    index -- index of the parallel process, for easy storage and
        retrieval of the result
    """

    CF = np.load('files/CF.npy', allow_pickle=True)
    FC = np.load('files/FC.npy', allow_pickle=True)
    
    CC = CF[start:end, :]@FC[:, :]
    
    with open(f'files/CC{index}.npy', 'wb') as f:
        np.save(f, CC)

def get_distance(matrix, rounds):
    """ Given the transition matrix, return the TV distance from the
    stationary distribution.
    The matrix and the stationary distribution are scaled to have integer entries

    The scale of 'matrix' is factor
    The scale of 'stationary_distribution' is A
    """
    # scale of matrix
    factor = (COMMON_DENOMINATOR*255**12)**rounds
    # scale of stationary_distribution
    A = 2**128 - 1
    distribution = FC[0, :]@matrix@CF
    
    # check that both distributions sum to 1
    if sum(distribution)*A != sum(stationary_distribution)*factor:
        return 1

    total_difference = 0
    for i in range(F):
        total_difference += abs(distribution[i]*A - stationary_distribution[i]*factor)

    # return log_2 of the distance
    return math.log2(total_difference)-math.log2(factor*A)-math.log2(2)

# COMPUTE CF, FC matrices

def compute_CF_matrix(C, F):
    """ Compute the CF matrix, that maps compressed layouts to full layouts """
    CF = np.zeros((C, F), dtype=object)

    for compressed_index in tqdm.tqdm(range(C)):
        compressed_layout = get_compressed_layout_by_index(compressed_index+1, k)
        
        total_layouts = 1
        for col in compressed_layout:
            total_layouts *= math.comb(4, col)
        
        for full_index in range(F):
            valid = True
            
            full_layout = get_layout_by_index(full_index+1, k, K)
            
            for i, col in enumerate(compressed_layout):
                if sum(full_layout[:, i]) != col:
                    valid = False
                    
            if valid:
                CF[compressed_index, full_index] = COMMON_DENOMINATOR // total_layouts

    # Save CF matrix
    with open('files/CF.npy', 'wb') as f:
        np.save(f, CF)

    return CF

def compute_FC_matrix(C, F):
    """ Compute the CF matrix, that maps compressed layouts to full layouts """
    FC = np.zeros((F, C), dtype=object)
    for full_index in tqdm.tqdm(range(F)):
        full_layout = get_layout_by_index(full_index+1, k, K)
        full_SR_layout = shift_rows(full_layout, k)
        
        for compressed_index in range(C):
            compressed_layout = get_compressed_layout_by_index(compressed_index+1, k)
            prob = 1
            
            for i, col in enumerate(compressed_layout):
                prob *= get_column_transition_probability(sum(full_SR_layout[:, i]), col, 255) * math.comb(4, col)
                    
            FC[full_index, compressed_index] = prob

    # Save FC matrix
    with open('files/FC.npy', 'wb') as f:
        np.save(f, FC)

    return FC

def compute_CC_matrix():
    """ Compute CC matrix, defined as CF@FC """
    # compute matrix multiplication in parallel
    # the matrix has 624 rows, so we will split them into groups of 50,
    # the last group gets 550-624

    processes = []
    for i, end in enumerate(range(50, 551, 50)):
        process = Process(target=multiply, args=(end-50, end, i))
        process.start()
        processes.append(process)
    # append last process
    last_process = Process(target=multiply, args=(550, 624, 11))
    last_process.start()
    processes.append(last_process)

    for process in processes:
        process.join()

    # collect intermediate matrix CC
    CC_list = []
    for i in range(12):
        CC_temp = np.load(f'files/CC{i}.npy', allow_pickle=True)
        CC_list.append(CC_temp)

    CC = np.vstack(CC_list)

    # fix mpf to int
    for i in range(CC.shape[0]):
        for j in range(CC.shape[1]):
            if type(CC[i, j]) == ctx_mp_python.mpf:
                print(i, j)
                break

    with open('files/CC.npy', 'wb') as f:
        np.save(f, CC)
    
    return CC

def verify_close(idx, rounds, matrix, limit):
    """ Verifies that the distance from pairwise is at most 2**(-limit)
    if we start from layout with index idx.
    
    Function outputs the indices of the layouts that are not close enough in
    files/violate_{rounds}_{limit}.txt.
    """

    CF = np.load('files/CF.npy', allow_pickle=True)
    FC = np.load('files/FC.npy', allow_pickle=True)
    
    C = 5**4 - 1
    F = 2**16 - 1
    
    stationary_distribution = np.ones(F, dtype=object)

    for full_index in range(F):
        layout_weight = bin(full_index+1).count('1')
        stationary_distribution[full_index] = 255**layout_weight
    
    factor = (COMMON_DENOMINATOR*255**12)**rounds
    A = 2**128 - 1
    distribution = FC[idx, :]@matrix@CF
    
    if sum(distribution)*A != sum(stationary_distribution)*factor:
        with open(f'files/violate_{rounds}_{limit}.txt', 'a') as f:
            f.write(f'incorrect normalization with idx {idx}\n')
        return

    total_difference = 0
    for i in range(F):
        total_difference += abs(distribution[i]*A - stationary_distribution[i]*factor)

    if math.log2(total_difference)-math.log2(factor*A)-math.log2(2) > -limit:
        with open(f'files/violate_{rounds}_{limit}.txt', 'a') as f:
            f.write(f'idx {idx} is at distance more than 2^{-limit}\n')

if __name__ == "__main__":

    mp.dps = 100
    k = 4
    K = 16

    C = 5**4 - 1 # number of compressed layouts
    F = 2**16 - 1 # number of full layouts

    """
    (README): change precomputed to True to avoid computing the matrices CC, CF, FC from scratch
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
        CF = compute_CF_matrix(C, F)
        FC = compute_FC_matrix(C, F)
        CC = compute_CC_matrix()

    max_power = 10

    # compute powers of CC
    print('Computing the powers of the transition matrix...', end="")
    CC_powers = [CC]
    for i in range(2, max_power+1):
        CC_powers.append(CC_powers[-1]@CC)
    print('DONE')

    # compute stationary distribution
    stationary_distribution = np.ones(F, dtype=object)

    for full_index in range(F):
        layout_weight = bin(full_index+1).count('1')
        stationary_distribution[full_index] = 255**layout_weight

    """
    (README)
    Below you can find the code for the 3 claims we make.
    1. Table 2: Statistical distance from pairwise independence of the r-round SPN
        with AES mixing and random S-boxes, given two inputs that differ in exactly
        one coordinate. This corresponds to starting from a layout I with Hamming
        weight 1.
    2. Theorem 5. The 7-round AES* is 2^{-128}-close to pairwise independent.
    3. Lemma 15. The 3-round AES* is 2^{-23.42}-close to pairwise independent.
    """






    """ 1. Table 2 / Plot distances from layout with Hamming weight 1 """

    x_list = []
    y_list = []
    table_list = []
    for i in range(max_power):
        rounds = i+2 # CC corresponds to 2 rounds, CC^2 to 3 etc
        distance = get_distance(CC_powers[i], rounds)
        
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




    """ 2. Theorem 5. Verify 7 rounds are 2^{-128} close """
    # If there is any layout 
    for i in tqdm.tqdm(range(0, F, 10)):
        # run 10 layouts in parallel and verify that 7 rounds are at least 2^{-128}-close to pairwise
        processes = []
        
        for r in range(10):
            if i + r < F:
                processes.append(Process(target=verify_close, args=(i+r, 7, CC_powers[5], 128.0)))
                processes[-1].start()
        
        processes[-1].join()
    
    with open(f'files/violate_{7}_{128.0}.txt', 'a') as f:
        f.write(f'DONE\n')







    """ 3. Lemma 15. Verify 3 rounds are 2^{-23.42} close """
    for i in tqdm.tqdm(range(0, F, 10)):
        # run 10 layouts in parallel and verify that 3 rounds are at least 2^{-23.42}-close to pairwise
        processes = []
        
        for r in range(10):
            if i + r < F:
                processes.append(Process(target=verify_close, args=(i+r, 3, CC_powers[1], 23.42)))
                processes[-1].start()
        
        processes[-1].join()

    with open(f'files/violate_{3}_{23.42}.txt', 'a') as f:
        f.write(f'DONE\n')
