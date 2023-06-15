import numpy as np

# AES plaintext is arranged as a 4x4 matrix of elements of F_{256}.
k = 4
k_square = 16

# The number of full layouts is 2^{k_square} - 1 (all possible subsets of kxk
# elements) except the empty set
F = 2**k_square - 1
# The number of compressed layouts is (k+1)^k - 1 (each one of the k columns
# has Hamming weight 0 to k), except the empty set
C = (k+1)**k - 1

# Compressed layout: list of length k that stores at the ith position
#   the Hamming weight of the ith layout column.
#                               [1 0 0 1]
# e.g. the compressed layout of [1 0 1 1] is [4 1 2 3].
#                               [1 1 1 1]
#                               [1 0 0 0]
# Note that two distinct layouts can map to the same compressed layout.
# A Hamming weight of 0, 1, 2, 3, 4 maps to
#                     1, 4, 6, 4, 1 distinct layout columns respectively.
# To avoid precision errors, we will scale the entries of the CF matrix
#   to become integers. The original entries of CF are fractions with the
#   number of layouts in a compressed layout in the denominator. A common
#   multiple of these denominators is LCM(1, 4, 6, 4, 1)^4 = 20736.
CF_SCALE = 20736

# The transition probabilities of Lemma 11 (TODO fix ref) have powers of
# 255 as a denominator. We scale the probabilities by 255 to keep the entries
# of our transition matrices as integers.
PROB_SCALE = 255

# The total scaling of the FC matrix is (PROB_SCALE**3)**k = PROB_SCALE**12
FC_SCALE = PROB_SCALE**12

# Scale of CC matrix (which is the product of CF and FC) is their product
CC_SCALE = CF_SCALE*FC_SCALE


""" Compute the column transition probability.
    Uses the exact formula from Lemma 11 of Section 6 and substitutes k = 4
    (TODO) update the number of the Lemma

    Keyword arguments:
    start_weight -- (Hamming) weight of the starting layout
    end_weight -- (Hamming) weight of the ending layout
    scale -- scales the probability to make it a whole number
        and avoid precision errors accummulating. The total scale is scale**3
"""
def get_column_transition_prob(start_weight, end_weight, scale=PROB_SCALE):
    if start_weight == 1:
        if end_weight == 4:
            return scale**3
        else:
            return 0
        
    elif start_weight == 2:
        if end_weight == 3:
            return scale**2
        elif end_weight == 4:
            return scale**3 - 4*scale**2
        else:
            return 0
        
    elif start_weight == 3:
        if end_weight == 2:
            return scale
        elif end_weight == 3:
            return scale**2 - 4*scale
        elif end_weight == 4:
            return scale**3 - 4*scale**2 + 10*scale
        else:
            return 0
        
    elif start_weight == 4:
        if end_weight == 1:
            return 1
        elif end_weight == 2:
            return scale - 4
        elif end_weight == 3:
            return scale**2 - 4*scale + 10
        elif end_weight == 4:
            return scale**3 - 4*scale**2 + 10*scale - 20
        else:
            return 0
        
    elif start_weight == 0:
        if end_weight == 0:
            return scale**3
        else:
            return 0
    else:
        return -1






""" Return the index of the layout.

    Keyword arguments:
    layout -- layout arranged as a kxk array of bits
"""
def get_layout_index(layout):
    # flatten layout
    flat_layout = layout.flatten()
    # interpret layout as a binary number
    index = 0
    for i in range(k_square):
        index *= 2
        index += flat_layout[i]
        
    # correct index to account for the fact that
    # the all-zeros layout is not valid
    return index - 1







""" Return the index of the compressed layout.

    Keyword arguments:
    compr_layout -- compressed layout arranged as a list of k
        numbers from 0 to k+1
"""
def get_compr_layout_index(compr_layout):
    # interpret layout as a number in base (k+1)
    index = 0
    for i in range(len(compr_layout)):
        index *= (k+1)
        index += compr_layout[i]
    
    # correct index to account for the fact that
    # the all-zeros layout is not valid
    return index - 1





""" Return the layout given its index
    The indexing just views the layout as a binary string

    Keyword arguments:
    index -- index of the layout we want to obtain
"""
def get_layout_by_index(index):
    # index starts from 0, but the all-zeros layout is not valid
    _index = index +  1

    C = np.zeros((k, k)).astype(int)
    for i in range(k_square):
        C[k-1-i//k, k-1-i%k] = _index%2
        _index //= 2
    return C







""" Return the compressed layout given its index
    The indexing just views the layout as a number in base k+1

    Keyword arguments:
    index -- index of the layout we want to obtain
"""
def get_compr_layout_by_index(index):
    # index starts from 0, but the all-zeros layout is not valid
    _index = index +  1

    compr_layout = []
    for i in range(k):
        compr_layout.append(_index % (k+1))
        _index //= (k+1)
    return compr_layout[::-1]






""" Apply the ShiftRows operation on the layout

    Keyword arguments:
    layout -- layout arranged as a kxk array of bits
"""
def shift_rows(layout):
    for i in range(k):
        layout[i, :] = list(layout[i, i:]) + list(layout[i, :i])
    return layout
