#%%
import pandas as pd
import datetime


def format_ds(x):
        '''
        Format colums ds to YY-MM-DD
        '''
        ts = pd.to_datetime(str(x))
        ts = ts.strftime('%Y-%m-%d')
        return ts

def format_ds_h(x):
        '''
        Format colums ds to YY-MM-DD hh-mm-ss
        '''
        ts = x#pd.to_datetime(str(x))
        ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        return ts
# %%
