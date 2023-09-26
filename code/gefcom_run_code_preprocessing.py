
#import pyreadr
import os
import pandas as pd
from utils.utils import format_ds_h
from preprocessing.preprocessing import Preprocessing 
'''
Programa para gerar o dataframe com as folhas para ser usado no calculo
dos clusters . Dataset do FGEFCOM2017
'''
print (os.getcwd())
RAW_DIR = '../../../Data_MHTS/'
RAW_DIR_GEF = RAW_DIR+'GEFCom2017/GEFCom2017-Final-Match_201903001613335637/'
RAW_FILE = 'hierarchy.xlsx'
#result = pyreadr.read_r(RAW_DIR+'gefcom.rda') # also works for Rds, rda

#ler 
#gefcom2017.pkl

gefcom = Preprocessing(RAW_DIR_GEF, RAW_FILE) 

merged_df = gefcom.load_preprocess_Gef(load_file='load.xlsx', hierarq_file='hierarchy.xlsx')
print ("merge_df\n", merged_df.head(2))

dir = 'data/processed/gefcom2017/'
file_name = 'gefcom2017.pkl'
merged_df.to_pickle(dir+file_name)

#gerar o gefcom2017_Y_df_bottom.pkl

spec_bottom_gefcom = [[ 'Level0', 'Level1', 'Level2', 'bottom']]
Y_df_bottom_gefcom, S_df_gefcom, tags_gefcom = gefcom.aggregate_df(merged_df, spec_bottom_gefcom)
print ("Y_df_bottom_gefcom\n", Y_df_bottom_gefcom.head(2))
#save to file
# format do df : unique_id, ds, y
file_name= 'gefcom2017_Y_df_bottom.pkl'
Y_df_bottom_gefcom.to_pickle(dir+file_name)

#gerar o df_cluster_sample e salvar a partir do gefcom2017_Y_df_bottom.pkl
Y_df_bottom_gefcom['ds'] = Y_df_bottom_gefcom['ds'].apply(format_ds_h)
Y_df_bottom_pivot_gefcom=Y_df_bottom_gefcom.pivot_table(index='unique_id',columns='ds',values='y')
Y_df_bottom_pivot_gefcom.reset_index(inplace=True)

print ("Y_df_bottom_pivot_gefcom\n", Y_df_bottom_pivot_gefcom.head(2))

#save to file format to clustering
file_name= 'gefcom2017_Y_df_bottom_pivot_df_cluster_sample.pkl'
Y_df_bottom_pivot_gefcom.to_pickle(dir+file_name)



#####################



