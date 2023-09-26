
import pandas as pd
import numpy as np
import os
import sys
import pickle
from cluster.clustering import Clustering

''''
programa para testar execucao

'''
print (os.getcwd())
DATA_CLUSTER_DIR = 'data/cluster/'


pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_similarity_matrix_ensemble.pkl' 
#load cluster   
cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
dic_cluster_dtw_ens = cluster.load_cluster()

print ("keys: ",dic_cluster_dtw_ens.keys() )
selected_clusters_dtw_ens, stats_df = cluster.select_clusters(dic_cluster_dtw_ens, sil_meas_thr=0.78 )

print (selected_clusters_dtw_ens)
print (stats_df)
#%%
dic_cluster_dtw_ens_new = {chave: dic_cluster_dtw_ens[chave] for chave in selected_clusters_dtw_ens}

print (dic_cluster_dtw_ens_new.keys())


def funcao():

    '''
    funcao que recebe arquivo e retorna menores valores de rmse
    '''

    eval_H_cluster_euc_ens_sil_9 = pd.read_pickle('data/evals_tmp/selected/evaluation_H_cluster_euclidean_ensemble_sil_9clusters.pkl')

    mask1 = eval_H_cluster_euc_ens_sil_9.index.get_level_values('metric').isin(['rmse'])
    eval_H_cluster_euc_ens_sil_9 = eval_H_cluster_euc_ens_sil_9.loc[mask1]
    mask2 = eval_H_cluster_euc_ens_sil_9.index.get_level_values('level')    
    mask2=[x for x in mask2 if 'H_Cluster' in x]
    
    df2 = eval_H_cluster_euc_ens_sil_9.loc[mask2]
    min_value = df2['MinTrace(mint_shrink)'].min()
    df2_min = df2[df2['MinTrace(mint_shrink)']==min_value]

    return df2_min

#funcao para gerar rsult com selecao ou all
# falta fazer padrao para o caso sem ensemble
def result_H_cluster_selection(eval_dir_all, error_metric, all='', seleuc='', seldtw=''):
    '''
    return df result 
    '''
    # Clusters euclidean - All
    if all:
        padrao_clu = 'H_cluster_euclidean_'+all
    elif sel:
        padrao_clu = 'H_cluster_euclidean_'+seleuc
    df1_ = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
    # Clusters euclidean with ensemble - All
    # ens silhouette
    padrao_clu = 'H_cluster_euclidean_ensemble_sil'
    df2_sil = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
    #ens Frequency
    padrao_clu = 'H_cluster_euclidean_ensemble_freq'
    df2_freq = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
    
    ### Clusters dtw
    if all:
        padrao_clu = 'H_cluster_dtw_'+all
    elif seldtw:
        padrao_clu = 'H_cluster_dtw_'+seldtw

    df3_ = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)       #['H_Cluster8_All']]
    # Clusters dtw with ensemble - All
    # ens silhouette
    padrao_clu = 'H_cluster_dtw_ensemble_sil'
    df4_sil = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)    #2.loc[['H_Cluster17_All']]
    # ens Frequency
    padrao_clu = 'H_cluster_dtw_ensemble_freq'
    df4_freq = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)

    result_eval_selection = pd.concat([df1_, df2_sil, df2_freq, 
                                 df3_, df4_sil, df4_freq, 
                                ], axis=0) 
    return result_eval_selection   


#%% 20/09/2023

# Create two sample DataFrames
data1 = {'Key': ['A', 'A','B', 'C'], 'Value1': [1, 1, 2, 3]}
data2 = {'Key': ['A', 'B', 'C'], 'Value2': ['X', 'Y', 'Z']}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Use pd.merge() to concatenate based on the 'Key' column
merged_df = pd.merge(df1, df2, on='Key', how='outer')

# %%
df1=pd.concat([load_teste.head(5),load_teste[load_teste['meter_id']==2].head(3) ])
df2=df_hierarq.head(2)
merged_df = pd.merge(df1, df2, on='meter_id', how='outer')
# %%
def load_preprocess_Gef(raw_dir,hierarq_file, load_file):

    '''
    receives the gefcom2017 load excel format
    meter_id	date	    h1	h2 ... h24
    1           yy-mm-dd    v1  v2 ... v24
    returns in format to clustering and prediction    
    meter_id	load	ds
    1           v1      yy-mm-dd h1
    '''
    df_hierarq = pd.ExcelFile(raw_dir+hierarq_file).parse('Sheet1')
    df = pd.ExcelFile(raw_dir+load_file).parse('Sheet1')
    df = df.melt(id_vars=df.columns[0:2], var_name='hora', value_name='load')
    df['hora']=df['hora'].str.replace(r'h(\d+)', r'\1', regex=True )
    df['hora'] = pd.to_numeric(df['hora'], errors='coerce')
    df['hora'] = pd.to_datetime(df['hora'], format='%H', errors='coerce')
    df['hora'] = df['hora'].dt.strftime('%H:%M:%S')
    df['ds']=df['date'].astype(str)+ ' ' +df['hora']

    df.drop(columns=['date', 'hora'], inplace = True)
    df['ds'] = pd.to_datetime(df['ds'])
    merged_df = pd.merge(df_hierarq, df, on='meter_id', how='outer')
    del df, df_hierarq
    merged_df = merged_df.rename({'load': 'y', 'mid_level': 'Level1', 'aggregate':'Level2', 'meter_id':'Level3' }, axis =1)
    merged_df.insert(0, 'Level0', 'Total')
    merged_df = merged_df[['Level0', 'Level1', 'Level2', 'Level3', 'ds', 'y']]

    return merged_df

# %%
# Create two sample DataFrames
data1 = {'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]}
data2 = {'Key': ['A', 'B', 'D', 'C'], 'Value2': ['X', 'Y', 'Z', 'W']}

df1t = pd.DataFrame(data1)
df2t = pd.DataFrame(data2)
#%%
# Merge the DataFrames based on 'Key' with an inner join
merged_df = df2t[df2t['Key'].isin(df1t['Key'])].merge(df1t, on='Key', how='inner')

# %%
