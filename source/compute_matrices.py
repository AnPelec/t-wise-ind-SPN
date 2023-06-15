import tqdm
from functools import *
from multiprocess import Process

import math
import numpy as np
from mpmath import *

from utils import *

""" Helper function for the parallel matrix multiplication.
    Computes CC[start:end, :], where CC := CF@FC
    
    Keyword arguments:
    start -- start of the range
    end -- end of the range
    index -- index of the parallel process, for easy storage and
        retrieval of the result
"""
def multiply(start, end, index):

    CF = np.load('files/CF.npy', allow_pickle=True)
    FC = np.load('files/FC.npy', allow_pickle=True)
    
    CC = CF[start:end, :]@FC[:, :]
    
    with open(f'files/CC{index}.npy', 'wb') as f:
        np.save(f, CC)



""" Function that computes the CF transition matrix.
    Recall that CF expands the compressed layouts to the full layout space.
    In particular, the total probability mass of a compressed layout is
    evenly distributed to the full layouts it contains.

    CF[i, j] = 0 if ith compressed layout does not contain jth full layout
             = 1/(# full layouts in ith compressed layout) o.w.

    To keep the entries integers, we scale everything by CF_SCALE

    Keyword arguments:
    C -- number of compressed layouts
    F -- number of full layouts
"""
def compute_CF_matrix(C, F):
    
    CF = np.zeros((C, F), dtype=object)

    # iterate over all compressed layouts by index
    for compr_index in tqdm.tqdm(range(C)):
        compr_layout = get_compr_layout_by_index(compr_index)
        
        # compute number of full layouts in compr_layout
        total_full_layouts = 1
        for col in compr_layout:
            total_full_layouts *= math.comb(4, col)
        
        # populate row by checking if a full_layout is in compr_layout
        for full_index in range(F):
            layout_in_compr_layout = True
            full_layout = get_layout_by_index(full_index)
            
            for i, col in enumerate(compr_layout):
                if sum(full_layout[:, i]) != col:
                    layout_in_compr_layout = False
                    
            if layout_in_compr_layout:
                CF[compr_index, full_index] = CF_SCALE // total_full_layouts

    # Save CF matrix
    with open('files/CF.npy', 'wb') as f:
        np.save(f, CF)

    return CF



""" Function that computes the FC transition matrix.
    Recall that FC maps a full layout to a compressed layout via the following
    sequence of operations:
        1. SR (ShiftRows)
        2. P (project a full layout to its compressed layout)
        3. MC (MixColumns - the compressed layout contains sufficient
            information to determine the exact transition probabilities)

    Keyword arguments:
    C -- number of compressed layouts
    F -- number of full layouts
"""
def compute_FC_matrix(C, F):

    FC = np.zeros((F, C), dtype=object)

    # iterate over all full layouts by index
    for full_index in tqdm.tqdm(range(F)):
        full_layout = get_layout_by_index(full_index)

        # 1. apply SR to get the first intermediate full layout
        inter_layout1 = shift_rows(full_layout)
        
        # 2. apply the projection to the compressed layout
        inter_layout2 = [sum(inter_layout1[:, i]) for i in range(k)]

        # 3. iterate over compressed layouts and compute the (scaled)
        # probability that inter_layout2 maps them after MC
        for compr_index in range(C):
            compr_layout = get_compr_layout_by_index(compr_index)
            scale_prob = 1
            
            # we multiply by math.comb(4, col) because the transition prob
            # is for one layout with the given Hamming weight. The compressed
            # layout contains math.comb(4, col) such layouts.
            for i, col in enumerate(compr_layout):
                scale_prob *= (get_column_transition_prob(inter_layout2[i],
                                                          col)
                             * math.comb(4, col)
                              )
                    
            FC[full_index, compr_index] = scale_prob

    # Save FC matrix
    with open('files/FC.npy', 'wb') as f:
        np.save(f, FC)

    return FC


""" Compute CC matrix, defined as CC := CF@FC
    CC maps a compressed layout to a compressed layout during the
    intermediate rounds of AES*.

    Implementation details:
     - It computes the matrix in parallel
     - The CC matrix is (5^4-1)x(5^4-1), i.e. 624 x 624. We split the rows
       of CC into groups of 50 and compute CF[i:i+50,:]@FC[:,:]
    - The last group gets 550 - 624
    - (TODO) parametrize the number of processes and their size
"""
def compute_CC_matrix():

    # initialize first 11 processes
    processes = []
    for i, end in enumerate(range(50, 551, 50)):
        process = Process(target=multiply, args=(end-50, end, i))
        process.start()
        processes.append(process)
    # append last process
    last_process = Process(target=multiply, args=(550, 624, 11))
    last_process.start()
    processes.append(last_process)

    # wait for processes to finish
    for process in processes:
        process.join()

    # collect intermediate matrix CC
    CC_list = []
    for i in range(12):
        CC_temp = np.load(f'files/CC{i}.npy', allow_pickle=True)
        CC_list.append(CC_temp)

    CC = np.vstack(CC_list)

    with open('files/CC.npy', 'wb') as f:
        np.save(f, CC)
    
    return CC
