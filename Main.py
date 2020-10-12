import dask.dataframe as dd
import os
import pandas as pd
import re
import ast
from typing import List
import time
import numpy as np
from itertools import chain

HOSTS_REGEX_PATTERN = r"^\[(('[a-z0-9\.]+'),{0,1})+\]$"
HOSTS_COM_REGEX_PATTERN = r"^\[(('[a-z0-9]+\.com'),{0,1})+\]$"
VALUES_REGEX_PATTERN = r"^\[([0-9\.]+,{0,1})+\]"

def chainer(s):
    return list(chain.from_iterable(s))

def list_size(data):
    return len(data[1:-1].split(','))

def save_process_output_data(dataframe) -> None:
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dataframe.to_csv(dir_path + '/results/result_gen.csv', index=False)
    except IOError:
        print("Could not save file")
        return

if __name__ == "__main__":
    start = time.perf_counter()
    # Read csv file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + "/resources/generated_input_1000.csv"
    df = pd.read_csv(path)
    
    # Convert to Dask
    ddata = dd.from_pandas(df, npartitions=8)
    
    # Clean dataframe - remove malformed data
    ddata = ddata[(ddata['hosts'].str.contains(HOSTS_REGEX_PATTERN) & 
                    ddata['values'].str.contains(VALUES_REGEX_PATTERN))]
    
    
    # Clean dataframe - remove incomplete data
    ddata['hosts_len'] = ddata.hosts.apply(lambda x: list_size(x), meta=('hosts', 'str'))
    ddata['values_len'] = ddata['values'].apply(lambda x: list_size(x), meta=('values', 'str'))
    
    ddata = ddata[(ddata['hosts_len']) == ddata['values_len']]
    
    ddata = ddata.compute()
    
    clean = time.perf_counter()
    print(f'Took {clean-start} to clean 1000000 rows of data')
  
    # Aggregate data
    ddata = dd.from_pandas(ddata, npartitions=8)
    ddata['hosts'] = ddata.hosts.apply(lambda x: ast.literal_eval(x), meta=('hosts', 'str'))
    ddata['values'] = ddata['values'].apply(lambda x: ast.literal_eval(x), meta=('values', 'str'))
    ddata = ddata.drop('hosts_len', 1)
    ddata = ddata.drop('values_len', 1)
    
    ddata = ddata.compute()
    
    res = pd.DataFrame({'hosts': chainer(ddata['hosts']),
                        'values': chainer(ddata['values'])})
    
    res = res.groupby('hosts')['values'].apply(list).reset_index(name='values')
    
    res['Min'] = res['values'].apply(lambda x: min(x))
    res['Max'] = res['values'].apply(lambda x: max(x))
    res['Avg'] = res['values'].apply(lambda x: sum(x)/len(x))
    res['Sum'] = res['values'].apply(lambda x: sum(x))
    
    res = res.drop('values', 1)

    agg = time.perf_counter()
    print(f'Took {agg-clean} to aggregate 1000000 rows of data')
    
    # Save data
    save_process_output_data(res)
    end = time.perf_counter()
    print(f'Took TOTAL {end-start} to complete 1000000 rows of data')