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

k = 4 # even though AES has 16 blocks, its mixing is full-branch
      # when viewed as a linear mapping from F^4 -> F^4 (i.e. between columns)
K = 16
# a compressed layout stores the Hamming weight of the layout columns
# Hence each Hamming weight can map to 1, 4, or 6 distinct layout columns
# For our matrices to have integer entries, we multiply everything by
# LCM(4, 6)^4 = 20736
COMMON_DENOMINATOR = 20736

def get_column_transition_probability(start_weight, end_weight, denominator):
    """Compute the column transition probability.
    Uses the exact formula from Lemma 11 of Section 6 and substitutes k = 4

    Keyword arguments:
    start_weight -- (Hamming) weight of the starting layout
    end_weight -- (Hamming) weight of the ending layout
    denominator -- scales the probability to make it a whole number
        and avoid precision errors accummulating
    """

    if start_weight == 1:
        if end_weight == 4:
            return denominator**3
        else:
            return 0
        
    elif start_weight == 2:
        if end_weight == 3:
            return denominator**2
        elif end_weight == 4:
            return denominator**3 - 4*denominator**2
        else:
            return 0
        
    elif start_weight == 3:
        if end_weight == 2:
            return denominator
        elif end_weight == 3:
            return denominator**2 - 4*denominator
        elif end_weight == 4:
            return denominator**3 - 4*denominator**2 + 10*denominator
        else:
            return 0
        
    elif start_weight == 4:
        if end_weight == 1:
            return 1
        elif end_weight == 2:
            return denominator - 4
        elif end_weight == 3:
            return denominator**2 - 4*denominator + 10
        elif end_weight == 4:
            return denominator**3 - 4*denominator**2 + 10*denominator - 20
        else:
            return 0
        
    elif start_weight == 0:
        if end_weight == 0:
            return denominator**3
        else:
            return 0
    else:
        return -1

# UTILS

def get_layout_index(layout):
    """ Return the index of the layout.

    Keyword arguments:
    layout -- layout arranged as a 4x4 array of bits
    """
    flat_layout = layout.flatten()
    index = 0
    for i in range(flat_layout.shape[0]):
        index *= 2
        index += flat_layout[i]
        
    return index

def get_compressed_layout_index(compressed_layout):
    """ Return the index of the compressed layout.

    Keyword arguments:
    compressed_layout -- compressed layout arranged as a list of
        numbers from 0 to 4
    """
    index = 0
    for i in range(len(compressed_layout)):
        index *= 5
        index += compressed_layout[i]
        
    return index

def get_layout_by_index(index, k, K):
    """ Return the layout given its index
    The indexing just views the layout as a binary string

    Keyword arguments:
    index -- index of the layout we want to obtain
    """
    C = np.zeros((k, k)).astype(int)
    for i in range(K):
        C[k-1-i//k, k-1-i%k] = index%2
        index //= 2
    return C

def get_compressed_layout_by_index(index, k):
    """ Return the compressed layout given its index
    The indexing just views the layout as a number in base 5

    Keyword arguments:
    index -- index of the layout we want to obtain
    """
    compressed_layout = []
    for i in range(k):
        compressed_layout.append(index%5)
        index //= 5
    return compressed_layout[::-1]

def shift_rows(layout, k):
    """ Apply the ShiftRows operation on the layout """
    for i in range(k):
        layout[i, :] = list(layout[i, i:]) + list(layout[i, :i])
    return layout

"""
Verify matrices
"""

def verify_CF_matrix(CF):
    """ Verify CF matrix """
    invalid = False
    for i in range(C):
        if sum(CF[i]) != COMMON_DENOMINATOR:
            invalid = True
            break

    if not invalid:
        return True
    else:
        return False

def verify_FC_matrix(FC):
    """ Verify FC matrix """
    invalid = False
    for i in range(F):
        if sum(FC[i]) != 255**12:
            invalid = True
            break
            
    if not invalid:
        return True
    else:
        return False

"""
Small tests
"""

def small_tests(CC, CC_pow2, CC_pow8):
    for i in range(C):
        if sum(CC[i]) != COMMON_DENOMINATOR*255**12:
            print(i)
    print('done')

    for i in range(C):
        if sum(CC_pow2[i]) != (COMMON_DENOMINATOR*255**12)**2:
            print(i)
    print('done')

    for i in range(C):
        if sum(CC_pow8[i]) != (COMMON_DENOMINATOR*255**12)**8:
            print(i)
    print('done')