#!/usr/bin/env python3
import os, sys, getopt
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from argparse import ArgumentParser

TARGET_VALUES = set([0, 1, 2, 3])
NONZERO_VALUES = set([1, 2, 3])
OUTFILE = 'submit.csv'

def is_valid_csv(parser, path):
    COL_DTYPES = {
    'datetime' : 'object',
    'road_id' : 'int64',
    'road_km' : 'int64',
    'target' : 'int64'
    }
    DATES_COLS = ['datetime']

    if not os.path.exists(path):
        parser.error(f'No such file: {path}')
        return
    fname, ext = os.path.splitext(path)
    if ext != '.csv':
        parser.error(f'Expected a csv file: {path}')
        return
    try:
        df = pd.read_csv(path, dtype=COL_DTYPES, 
                parse_dates=DATES_COLS)
    except Exception as e:
        parser.error(f'Error converting to pandas.DataFrame: {str(e)}')
        return
    if not set(df.target.unique()).issubset(TARGET_VALUES):
        parser.error(f'Unexpected target values in file: {path}')
        return
    return df

def remove_zero(df):
    return (df[(~df['target'].isnull()) & (df['target']!=0)])

if __name__ == '__main__':    
    parser = ArgumentParser(description = 
            'Prepare file with answers to send')
    parser.add_argument('-f', 
            dest='file',
            required=True,
            type=lambda x: is_valid_csv(parser, x),
            help='path to csv file with true events')

    parser.add_argument('-o', 
            dest='outfile',
            default=OUTFILE,
            help='name and path to csv file to output results')

    args = parser.parse_args()
    remove_zero(args.file).to_csv(args.outfile, index = False)
    if os.path.exists(args.outfile):
        if (os.stat(args.outfile).st_size < 3145728):
            print(args.outfile,'successfully created')            
        else:
            print(args.outfile,'file size too large(<3mb)')
    
        
