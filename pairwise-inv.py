import copy
import tqdm
import random
import itertools

from tabulate import tabulate

import math
import scipy
import numpy as np
import pandas as pd
from decimal import *
from pyfinite import ffield

import seaborn as sns
from matplotlib import pyplot as plt

def field_pow(base, power, F):
    """ Compute base**power over field F """
    if power == 0:
        return 1
    elif power == 1:
        return base
    
    partial = field_pow(base, power//2, F)
    partial2 = F.Multiply(partial, partial)
    if power%2 == 0:
        return partial2
    else:
        return F.Multiply(partial2, base)

def INV(byte, F):
    """ Compute INV S-box """
    return field_pow(byte, n-2, F)

def get_INV_transition_matrix(m):
    """ Compute the INV transition matrix.
    Does not include 0 in the set of states.
    Matrix is unnormalized
    (have to divide by n, the size of F)
    
    Keyword arguments
    m -- field size is 2**m
    """
    
    n = 2**m
    F = ffield.FField(m)
    T = np.zeros((n-1, n-1), dtype=object)
    
    for x in range(1, n):
        # iterate over random key
        for key in range(n):
            y = F.Add(x, key)
            res = F.Add(INV(y, F), INV(key, F))
            # we transition from x to res
            # our index starts from 0
            T[x-1, res-1] += 1
            
    return T

def get_distance(u, v):
    """ Return log(d_{TV}(u, v))
    Vectors are not normalized.
    """
    # Compute scale of vectors
    scale_u = sum(u)
    scale_v = sum(v)
    
    total_dist = 0
    
    for elem_u, elem_v in zip(u, v):
        total_dist += abs(scale_v*elem_u - scale_u*elem_v)
        
    true_dist = math.log2(total_dist) - math.log2(scale_u) - math.log2(scale_v) - 1
    return true_dist

def get_INV_distance(transition_matrix):
    height, _ = transition_matrix.shape
    stationary = np.ones((height,), dtype=object)
    
    max_dist = -1000 # keep max log distance
    
    # iterate over all starting points
    # (differences between plaintexts, excluding 0)
    for i in range(height):
        curr_dist = get_distance(transition_matrix[i], stationary)
        max_dist = max(max_dist, curr_dist)
    
    return max_dist

if __name__ == "__main__":

    """
    (README)
    Below you can find the code to recover the entries of Table 3.
    Parameters:
    n -- the size of the field, we use n = 2^m, m = 8, since this is the
        field AES is defined over
    max_power -- compute the statistical distance of INV^{\otimes r} from a
        random S-box for r up to max_power
    """

    m = 8
    F = ffield.FField(m)
    n = 2**m

    T = get_INV_transition_matrix(m)
    max_power = 40

    powers_of_T = [T]
    for power in tqdm.tqdm(range(max_power)):
        powers_of_T.append(powers_of_T[-1]@T)
        
    table_list = []
    for i in range(max_power):
        table_list.append([i+1, get_INV_distance(powers_of_T[i])])

    print(tabulate(table_list, headers=['# repetitions', 'log_2(dist) to random']))