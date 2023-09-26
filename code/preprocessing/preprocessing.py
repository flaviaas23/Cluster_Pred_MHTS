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
        load and preprocess the tourism data, receives:
        Quarter	Region	    State	        Purpose	    Trips
        1998 Q1	Adelaide	South Australia	Business	135.077690
        returns:
        Country	    Region	    State	    Purpose	         ds	         y
        Australia	Adelaide	South Australia	Business	1998-01-01	135.077690
        '''   
        Y_df = pd.read_csv(self.raw_dir+self.raw_file)
        Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
        Y_df.insert(0, 'Country', 'Australia')
        Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
        Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    
        return Y_df

    # Preprocess gefcom2017 data
    def load_preprocess_Gef(self, load_file='load.xlsx', hierarq_file='hierarchy.xlsx' ):
        '''
        receives the gefcom2017 load excel format
        meter_id	date	    h1	h2 ... h24
        1           yy-mm-dd    v1  v2 ... v24
        returns in format to clustering and prediction    
        meter_id	load	ds
        1           v1      yy-mm-dd h1
        '''
        df_hierarq = pd.ExcelFile(self.raw_dir+hierarq_file).parse('Sheet1')

        df = pd.ExcelFile(self.raw_dir+load_file).parse('Sheet1')

        df = df.melt(id_vars=df.columns[0:2], var_name='hora', value_name='load')
        df['hora']=df['hora'].str.replace(r'h(\d+)', r'\1', regex=True )
        #df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
        #df['hora']=df['hora'].str.replace(r'24',r'00', regex=True )

        df['hora'] = df['hora'].apply(lambda x: x+':00:00')

        #df['hora'] = pd.to_datetime(df['hora'], format='%H', errors='coerce')
        #df['hora'] = df['hora'].dt.strftime('%H:%M:%S')
        df['ds']=df['date'].astype(str)+ ' ' +df['hora']
        df['ds']=df['ds'].str.replace(r'24:00:00',r'00:00:01', regex=True )

        df.drop(columns=['date', 'hora'], inplace = True)
        df['ds'] = pd.to_datetime(df['ds'])

        merged_df = df_hierarq[df_hierarq['meter_id'].isin(df['meter_id'])].merge(df, on='meter_id', how='inner')
        #merged_df = pd.merge(df_hierarq, df, on='meter_id', how='outer')
        del df, df_hierarq

        merged_df = merged_df.rename({'load': 'y', 'mid_level': 'Level1', 'aggregate':'Level2',  'meter_id':'bottom' }, axis=1)
        merged_df['bottom']=merged_df['bottom'].astype(str)
        merged_df.insert(0, 'Level0', 'Total')
        merged_df = merged_df[['Level0', 'Level1', 'Level2', 'bottom', 'ds', 'y']]
        
        return merged_df
 
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
    
    def save_to_pickle(df, dir,file_name):
        ''''
        save to pickle_file
        '''
        df.to_pickle(dir+file_name)

        return
    
    
        
       
        

# %%
