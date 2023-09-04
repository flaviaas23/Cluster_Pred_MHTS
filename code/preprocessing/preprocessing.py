#%%
import pandas as pd
import numpy as np

import pickle

#to aggregate data in hierarchies

from hierarchicalforecast.utils import aggregate

class Preprocessing:
    #def __init__(self, raw_dir='../../Data_MHTS/', raw_file='' ):
    def __init__(self, raw_dir, raw_file ):
        self.raw_dir = raw_dir #'../../../../Data_MHTS/'
        self.raw_file = raw_file #'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
        #self.pkl_file = ''
        self.col_unique = 'unique_id'

    
    # def save_dfToPickle(self, df):
    #     #function to salve to pickle file
    #     obj_pickle={}

    #     obj_pickle= {
    #         filename: df,     # df to be saved
    #          }

    #     with open(self.raw_dir+"/pickle_"+self.raw_file+"_df.pickle", 'wb') as handle:
            # pickle.dump(obj_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Preprocess data
    def load_preprocess_tourism(self):
        '''
        load and preprocess the tourism data
        '''   
        Y_df = pd.read_csv(self.raw_dir+self.raw_file)
        Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
        Y_df.insert(0, 'Country', 'Australia')
        Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
        Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    
        return Y_df
 
    def split_test_train(self, Y_df, h):
        '''
        split df in test and train using h
        '''
        Y_test_df = Y_df.groupby(self.col_unique).tail(h)
        Y_train_df = Y_df.drop(Y_test_df.index)

        Y_test_df = Y_test_df.set_index(self.col_unique)
        Y_train_df = Y_train_df.set_index(self.col_unique)

        return Y_test_df, Y_train_df

    def aggregate_df(self, Y_df , spec):
       '''
       agggregate data in hierarchies of spec, 
       returns Y_df aggregated, S_df and tags
       '''
       Y_df, S_df, tags = aggregate(Y_df, spec)
       Y_df = Y_df.reset_index()
       
       return Y_df, S_df, tags
    
    
        
       
        

# %%
