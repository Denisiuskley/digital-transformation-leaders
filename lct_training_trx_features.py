try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    get_ipython().run_line_magic('matplotlib', 'qt')
except:
    pass

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from tqdm import tqdm
import joblib

'''
Generate additional features of transactions.
1) Used grouped data of event type and month.
2) Create sparce pivot table of 'currency', "src_type11", "dst_type11". 
    Redused size with SVD algorithm
'''

def actions(table):
    table['client_id'] = table['client_id'] + '_month=' + table['month'].astype(str)
    table.drop(['month'], axis = 1, inplace = True)
    return table

def table_create(df, test = False):    
    grouped = df[df[cols_category[0]].isin(event_type)].groupby(['client_id'] + cols_category)['amount'].sum()
    table = grouped.unstack(level=1).fillna(0).astype(float).reset_index()
    if test:
        table = table.loc[table.groupby('client_id')['month'].idxmax()]
    else:
        table['client_id'] = table['client_id'] + '_month=' + table['month'].astype(str)
    table = table.drop(['month'], axis = 1)
    cols = ['client_id'] + [f'{cols_category[0]}_{col}' for col in list(table.columns)[1:]]
    table.columns = cols
    table['amount'] = table.drop(['client_id'], axis = 1).sum(axis = 1)
    return table

def pivot(df, drop):
    uns = []
    for i in tqdm(range(len(df) // drop + 1)): 
        uns.append(pd.pivot_table(df.iloc[i * drop: (i + 1) * drop], values='amount', columns=cols_category, index=['client_id', 'month'], aggfunc='mean', fill_value = 0))
    uns = pd.concat(uns)
    return uns

transactions_train = pd.read_parquet("Hackathon/trx_train.parquet")
transactions_test = pd.read_parquet("Hackathon/trx_test.parquet")

print('Dataset red')

# Change date to datetime format
transactions_train['event_time'] = pd.to_datetime(transactions_train['event_time'])
transactions_test['event_time'] = pd.to_datetime(transactions_test['event_time'])
transactions_train['month'] = transactions_train['event_time'].dt.month
transactions_test['month'] = transactions_test['event_time'].dt.month
transactions_train['year'] = transactions_train['event_time'].dt.year
transactions_test['year'] = transactions_test['event_time'].dt.year
transactions_train = transactions_train[transactions_train['year']==2022]

cols_category=["event_type", 'month']
event_type = list(transactions_train[cols_category[0]].unique())

# Create grouped data    
train = table_create(transactions_train, test = False)
val = table_create(transactions_test, test = False)
test = table_create(transactions_test, test = True)

train.to_parquet("created_data/train_trx_f.parquet", index=False, engine="pyarrow", compression="snappy")
val.to_parquet("created_data/val_trx_f.parquet", index=False, engine="pyarrow", compression="snappy")
test.to_parquet("created_data/test_trx_f.parquet", index=False, engine="pyarrow", compression="snappy")

cols_category=["event_subtype", 'month']
event_type = list(transactions_train[cols_category[0]].unique())

# Create grouped data    
train = table_create(transactions_train, test = False)
val = table_create(transactions_test, test = False)
test = table_create(transactions_test, test = True)

train.to_parquet("created_data/train_trx_f2.parquet", index=False, engine="pyarrow", compression="snappy")
val.to_parquet("created_data/val_trx_f2.parquet", index=False, engine="pyarrow", compression="snappy")
test.to_parquet("created_data/test_trx_f2.parquet", index=False, engine="pyarrow", compression="snappy")

cols_category=['currency', "src_type11", "dst_type11"]
scaler = MinMaxScaler()
n_comp = 10
tsvd = TruncatedSVD(n_components=n_comp, algorithm='arpack', random_state=42)

# Create pivot table by chunk for  
drop = 5_000_000
uns_train = pivot(transactions_train, drop)
uns_test = pivot(transactions_test, drop)
del transactions_train, transactions_test 
while gc.collect():
    gc.collect()

len_train = len(uns_train)
uns = pd.concat([uns_train, uns_test]).fillna(0)

uns_train = uns.iloc[:len_train]
uns_test = uns.iloc[len_train:]
del uns
while gc.collect():
    gc.collect()

index = uns_train.iloc[:, 0].reset_index()
index.columns = ['client_id', 'month', 'none']
index.drop(['none'], axis = 1, inplace = True)
uns_train = scaler.fit_transform(uns_train)

ts = tsvd.fit_transform(uns_train)
cols = [f'trx_sparse_tsvd_{i}' for i in range(n_comp)]

index[cols] = ts



index = actions(index)
index.to_parquet("created_data/train_trx_f3.parquet", index=False, engine="pyarrow", compression="snappy")

index = uns_test.iloc[:, 0].reset_index()
index.columns = ['client_id', 'month', 'none']
index.drop(['none'], axis = 1, inplace = True)
uns_test = scaler.transform(uns_test)

ts = tsvd.transform(uns_test)

index[cols] = ts


test = index.loc[index.groupby('client_id')['month'].idxmax()]
val = actions(index)

val.to_parquet("created_data/val_trx_f3.parquet", index=False, engine="pyarrow", compression="snappy")
test.drop(['month'], axis = 1).to_parquet("created_data/test_trx_f3.parquet", index=False, engine="pyarrow", compression="snappy")



















