from evaluate_models import ExperimentData
import sys
import pandas as pd
import numpy as np

df = ExperimentData(sys.argv[1]).pd_array
dataset = sys.argv[2]

df = df.loc[df['dataset'] == dataset]
df.drop_duplicates(inplace=True)


df_dek = df.loc[df['kernel'] == 'dek']
df_nnk = df.loc[df['kernel'] == 'nnk']
df_rbf = df.loc[df['kernel'] == 'rbf']
df_ntk = df.loc[df['kernel'] == 'ntk']

def merge_dek_scores(df):
    df_nonzero = df.loc[df['c_kernel'] == 'relu'].sort_values('seed')
    df_zero = df.loc[df['c_kernel'] == 'zero'].sort_values('seed')
    
    for s in df_zero['seed']:
        assert s in df_nonzero['seed'].values, str(s)

    for s in df_nonzero['seed']:
        assert s in df_zero['seed'].values, str(s)
    col_nonzero = df_nonzero['rmse'].to_numpy()
    col_zero = df_zero['rmse'].to_numpy()
    col = np.minimum(col_nonzero, col_zero)

    ret = df_nonzero.copy(deep=True).sort_values('seed')
    ret['rmse'] = col


    num_times_zero_wins = np.sum((col_zero - col_nonzero) < 0)

    return ret, num_times_zero_wins

def get_rmse_pm(df):
    return  '$' + "{:.4f}".format(df['rmse'].mean()) + \
    '\pm ' + "{:.4f}".format(df['rmse'].std()) + '$'

def print_row(df_dek, df_nnk, df_ntk, df_rbf,  zero_wins):
    num_dek_nnk = str(zero_wins)

    rmse_dek  = get_rmse_pm(df_dek) 
    rmse_nnk  = get_rmse_pm(df_nnk) 
    rmse_ntk  = get_rmse_pm(df_ntk) 
    rmse_rbf  = get_rmse_pm(df_rbf) 

    print(dataset + ' & ' + rmse_dek + ' & ' + rmse_nnk + ' & ' + rmse_ntk + ' & ' + rmse_rbf + ' & ' + num_dek_nnk + '\\\\ \hline')

def check_consistent(df1, df2, df3, df4):
    print('The data for the following seeds is missing:')
    for s in df1['seed']:
        if (not s in df2['seed'].values) or \
            (not s in df3['seed'].values) or\
            (not s in df4['seed'].values):
            print(s)

    for s in df2['seed']:
        if (not s in df3['seed'].values) or \
            (not s in df4['seed'].values) or \
            (not s in df1['seed'].values):
            print(s)

    for s in df3['seed']:
        if (not s in df1['seed'].values) or \
            (not s in df2['seed'].values) or \
            (not s in df4['seed'].values):
            print(s)

    for s in df4['seed']:
        if (not s in df1['seed'].values) or\
            (not s in df2['seed'].values) or \
            (not s in df3['seed'].values):
            print(s)

check_consistent(df_dek, df_nnk, df_rbf, df_ntk)
df_dek, zero_wins = merge_dek_scores(df_dek)

print_row(df_dek, df_nnk, df_ntk, df_rbf, zero_wins)

