#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ipdb as pdb
from tqdm import tqdm

def weight(j, df):
    return df.loc[j, 'weight']

def value(j, df):
    return df.loc[j, 'value']

def brute_force_it(input_data):
    lines = input_data.split('\n')
    lines = [line for line in lines if line != '']
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    ks_capacity = int(firstLine[1])
    lines_int = [map(int, x.split(' ')) for x in lines[1:]]
    items = np.array(lines_int)
    highest_value = 0
    for i in tqdm(range(2**item_count)):
        string = bin(i).lstrip('0b').zfill(item_count)
        mask = np.array(map(int, string))
        weight = (items[:, 1] * mask).sum()
        if weight < ks_capacity:
            total_value = (items[:, 0] * mask).sum()
            if total_value > highest_value:
                highest_value = total_value
                highest_mask = mask
    return highest_value, highest_mask



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    lines = [line for line in lines if line != '']
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    ks_capacity = int(firstLine[1])
    lines_int = [list(map(int, x.split(' '))) for x in lines[1:]]
    t = np.zeros([ks_capacity+1, item_count+1], dtype=np.int64)
    df = pd.DataFrame([[0, 0]]+ lines_int, columns=('value', 'weight'))  # add [0,0] to top of list
    for i in tqdm(range(ks_capacity+1)):
        for j in range(item_count+1):
            if weight(j, df) == 0:
                continue
            elif weight(j, df) > i:
                t[i][j] = t[i][j-1]
            else:
                t[i][j] = max(value(j, df) + t[i-weight(j, df)][j-1], t[i][j-1])
    taken = np.zeros([item_count], dtype=np.int8)
    i = ks_capacity
    total_value = 0
    for j in range(item_count, 0, -1):
        if t[i][j] == t[i][j-1]:
            taken[j-1] = 0
        else:
            taken[j-1] = 1
            i = i - weight(j, df)
            total_value += value(j, df)
    """
    dp = pd.DataFrame(np.zeros([ks_capacity+1, item_count+1], dtype=np.int8))
    for item_number, item in df.iterrows():
        for capacity, total_value in enumerate(dp[item_number]):
            print(item_number, item.value, item.weight, capacity, total_value)
            #if item.weight <= capacity:
                #dp.loc[capacity, item_number] = item.weight


    """
    """
    df['value_density'] = df['value'] / df['weight']
    df = df.sort_values(by='value_density', ascending=False)
    df['taken'] = 0
    remaining_capacity = ks_capacity
    for idx, row in df.iterrows():
        if row.weight <= remaining_capacity:
            df.loc[idx, 'taken'] = 1
            remaining_capacity -= row.weight
    #df['cumulative_weight'] = df['weight'].cumsum(axis=0)
    #df['taken'] = df['cumulative_weight'] <= ks_capacity
    #df['taken'] = df['taken'].apply(int)
    total_value = df[df['taken']==1]['value'].sum()
    df.sort_index(inplace=True)

    print(df)
    """
    # prepare the solution in the specified output format
    output_data = str(total_value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def get_input_data(file_location):
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    return input_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data = get_input_data(file_location)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
    print(solve_it(input_data))

