import tqdm
from tabulate import tabulate

import math
import numpy as np
from pyfinite import ffield

""" Compute base ** exponent over field F.
    Uses divide and conquer for efficiency.
"""
def field_pow(base, exponent, F):
    # base cases
    if exponent == 0:
        return 1
    elif exponent == 1:
        return base
    
    # compute intermediate result base ** (exponent//2)
    inter_power = field_pow(base, exponent//2, F)

    # square intermediate result
    inter_power_square = F.Multiply(inter_power, inter_power)

    # compute power depending on the parity of the exponent
    if exponent%2 == 0:
        return inter_power_square
    else:
        return F.Multiply(inter_power_square, base)

""" Compute INV S-box. This is equal to
    INV(x) = 1/x if x != 0
           = 0   if x == 0
    This is the same as x^{n-2}, where n is the size of the field.

    Keyword arguments:
    x -- element whose inverse we want to compute
    n -- size of the field, e.g. 2^8 for our application
    F -- field
"""
def INV(x, n, F):
    return field_pow(x, n-2, F)

""" Compute the INV transition matrix, where
    T[x, y] = Pr[S(difference x) = difference y]*n for x, y != 0
    The n factor is to keep all the entries integer and avoid precision errors
    
    Keyword arguments
    b -- field size is n := 2 ** b
"""
def get_INV_transition_matrix(b):
    
    n = 2 ** b
    F = ffield.FField(b)
    T = np.zeros((n-1, n-1), dtype=object)
    
    # iterate over initial dfference 
    for x in range(1, n):
        # iterate over random key
        for key in range(n):
            # compute difference after S-box
            inter_value = F.Add(x, key)
            y = F.Add(INV(inter_value, n, F), INV(key, n, F))
            # we transition from x to y (-1 to keep the matrix 0-index)
            T[x-1, y-1] += 1
            
    return T

""" Return the logarithm of the TV distance between distributions u and v,
    where u and v are represented as 1-D vectors of the same size.
    The vectors scaled to have integer entries.
"""
def get_distance(u, v):
    # Compute scale of vectors
    scale_u = sum(u)
    scale_v = sum(v)
    
    total_dist = 0
    for elem_u, elem_v in zip(u, v):
        total_dist += abs(scale_v*elem_u - scale_u*elem_v)
        
    true_dist = (math.log2(total_dist)
                 - math.log2(scale_u)
                 - math.log2(scale_v)
                 - 1)
    return true_dist

""" Compute the statistical distance from uniform after one step of the random
    walk that is represented by transition_matrix.

    Keyword arguments:
    transition_matrix -- square matrix that represents the walk, by storing
        the probability that we go from state i to state j at
        transition_matrix[i, j]
"""
def get_walk_distance(transition_matrix):
    # compute number of states
    num_states, _ = transition_matrix.shape

    # compute (scaled) uniform distribution
    stationary = np.ones((num_states,), dtype=object)
    
    # keep max log distance
    max_dist = -1000
    
    # iterate over all states (correspond to plaintext differences,
    # excluding 0)
    for i in range(num_states):
        curr_dist = get_distance(transition_matrix[i], stationary)
        max_dist = max(max_dist, curr_dist)
    
    return max_dist

""" Table 3: Statistical distance upper bound of (AES S-box) repeated r times
    from a pairwise random S-box over F_{2^8} \ {0}
    
    Keyword arguments:
    n -- the size of the field, we use n = 2^m, m = 8, since this is the
        field AES is defined over
    max_power -- compute the statistical distance of INV^{\otimes r} from a
        random S-box for r up to max_power
"""
if __name__ == "__main__":

    # setup parameters
    b = 8
    F = ffield.FField(b)
    n = 2 ** b

    # compute transition matrix
    T = get_INV_transition_matrix(b)

    # compute powers of transition matrix
    max_power = 40
    powers_of_T = [T]
    for power in tqdm.tqdm(range(max_power)):
        powers_of_T.append(powers_of_T[-1]@T)
        
    # compute distances of transition matrix walk from uniform
    table_list = []
    for i in range(max_power):
        table_list.append([i+1, get_walk_distance(powers_of_T[i])])

    # print results
    print(tabulate(table_list, headers=['# repetitions', 'log_2(dist) to random']))
