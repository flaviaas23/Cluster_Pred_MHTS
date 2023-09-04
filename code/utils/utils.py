import pandas as pd
import datetime

#%%
def format_ds(x):
        '''
        Format colums ds to YY-MM-DD
        '''
        ts = pd.to_datetime(str(x))
        ts = ts.strftime('%Y-%m-%d')
        return ts
# %%
