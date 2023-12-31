#%%
import pandas as pd
import numpy as np
import os
import sys
import glob

import pickle

''''
Programa para gerar tabela de avaliacao das predicoes das hierarquias
'''
#%%
def df_cols_to_numeric(df):
    ''''
    Set type to be numeric of eval columns 
    '''
    for x in df.columns[1:]:
        df[x] = pd.to_numeric(df[x])
        print (x)

    return df

#%%
def get_rows_from_cols(df, e_metric, cols, col_index_to_filter='level'):
    ''''
    From df returns row filtered
    cols=columns index values to be filtered
    '''
    print ('e_metric= , cols=, col_index_to_filter='.format(e_metric, cols, col_index_to_filter))
    print (df.head(1))
    mask1 = df.index.get_level_values('metric').isin([e_metric])
    df2 = df.loc[mask1]
    print (df2.head(1))
    #df2_ = df2.loc[[cols]]
    mask2 = df2.index.get_level_values(col_index_to_filter)
    mask2=[x for x in mask2 if cols in x]
    print ('mask2',mask2)
    df2_ = df2.loc[mask2]
    print (df2.head(1))
    return df2_

#### for clusters hierarchies
#%%
def get_small_metric_row(df, e_metric, both=0):
    ''''
    Receives a df and returns its row(s) with small column metric value
    '''
    #makes sure the errors columns values are numeric
    df = df_cols_to_numeric(df)

    # filter the metric to be considered
    mask1 = df.index.get_level_values('metric').isin([e_metric])
    df2 = df.loc[mask1]
    #para excluir o bottom e pegar a hierarquia , pego o segundo menor
    index_second = df2['MinTrace(mint_shrink)'].nsmallest(2).index[1]
    df2_= df2.loc[[index_second]]

    if both:
        df_=df.loc[[index_second]]
        return df2_, df_
    return df2_

#%%
def get_small_metric_row_ind(df, e_metric, col_index_to_filter='level',bottom=0, both=0):
    ''''
    20230817:Criei esta funcao para fazer um filtro do level para pegar os
    indices que tem Cluster numa parte deles
    depois pensar numa forma de fazer junto com a funcao acima
    Receives a df and returns its row with small column metric value
    '''
    #makes sure the errors columns values are numeric
    df = df_cols_to_numeric(df)

    # filter the metric to be considered
    
    mask1 = df.index.get_level_values('metric').isin([e_metric])
    df2 = df.loc[mask1]
    
    #to get all indexes
    mask2 = df2.index.get_level_values(col_index_to_filter)    
       
    if bottom:
        #if want to exclude just bottom
        mask2=[x for x in mask2 if "Bottom" not in x ]
    else:
        #if want to just leave clusters
        #mask2=[x for x in mask2 if 'Cluster' in x or 'Dominio' in x]
        mask2=[x for x in mask2 if 'Cluster' in x]

    df2 = df2.loc[mask2]
    min_value = df2['MinTrace(mint_shrink)'].min()
    df2_min = df2[df2['MinTrace(mint_shrink)']==min_value]

    #para excluir o bottom e pegar a hierarquia , pego o primeiro menor elemento
    #no caso de uso de clusters all
    # index_second = df2['MinTrace(mint_shrink)'].nsmallest(1).index[1]
    # df2_= df2.loc[[index_second]]

    if both:
        # so retorna as linhas iguais ao min value ...
        return df[df['MinTrace(mint_shrink)']==min_value]
    # if both:
    #     df_=df.loc[[index_second]]
    #     return df2_, df_
    return df2_min


#%%
#reading evaluation files
eval_dir='data/evals_tmp/'

padrao = os.path.join(eval_dir, 'evaluation_H*.pkl')

eval_files = glob.glob(padrao)
# %%
['../data/evals_tmp/evaluation_H_cluster_euclidean_All.pkl',
 '../data/evals_tmp/evaluation_H_dominio_cluster_dtw_All.pkl',
 '../data/evals_tmp/evaluation_H_dominio_cluster_euclidean_All.pkl',
 '../data/evals_tmp/evaluation_H_dominio_cluster_euclidean_ensemble_All.pkl',
 '../data/evals_tmp/evaluation_H_dominio.pkl',
 '../data/evals_tmp/evaluation_H_dominio_cluster_dtw_ensemble_All.pkl',
 '../data/evals_tmp/evaluation_H_cluster_euclidean_ensemble_All.pkl',
 '../data/evals_tmp/evaluation_H_cluster_dtw_All.pkl',
 '../data/evals_tmp/evaluation_H_cluster_dtw_ensemble_All.pkl']

#%%
def gen_df_row_dom(eval_dir, error_metric, padrao, cols):
    '''
    gets row with bets metric
    '''
    eval_H = read_files(eval_dir, padrao)
    print (eval_H)

    df = get_rows_from_cols(eval_H[0], error_metric, cols) #eval_H_dom2.loc[['H_Dominio']]

    return df 

#%%
def gen_df_row_cluster(eval_dir, error_metric, padrao):
    '''
    gets row with bets metric
    '''
    eval_H = read_files(eval_dir, padrao)
    #print (eval_H)

    df = get_small_metric_row(eval_H[0], error_metric) #H_Cluster13_All

    return df 


#%%
### Dominio only
'''
eval_file = [x for x in eval_files if 'H_dominio.' in x][0]
eval_H_dom = pd.read_pickle(eval_file)
eval_H_dom = df_cols_to_numeric(eval_H_dom)
'''
#%%
### Dominio only
error_metric= 'rmse'
padrao = 'H_dominio'
cols = 'H_Dominio'
df9_ = gen_df_row_dom(eval_dir, error_metric, padrao, cols)
#%%
### dominio with clusters euclidean
eval_dir_all=eval_dir+'All/'
padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_All'
df5_ = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_all, cols)
#%%
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom+cluster
df5_cl = gen_df_row_dom(eval_dir_all,error_metric, padrao_dom_clu_euc_all, 'H_Cluster')
df5_cl_min= get_small_metric_row_ind(df5_cl, error_metric)
#%%
### dominio with clusters euclidean ensemble
'''
eval_file = [x for x in eval_files if 'H_dominio_cluster_euclidean_ensemble_sil_All' in x][0]
eval_H_dom_cluster_euc_ensemble = pd.read_pickle(eval_file)
eval_H_dom_cluster_euc_ensemble = df_cols_to_numeric(eval_H_dom_cluster_euc_ensemble)
df6_ = get_rows_from_cols(eval_H_dom_cluster_euc_ensemble, error_metric,'H_Dominio') #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
'''
#%%
### dominio with clusters euclidean ensemble
padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_sil_All'
df6_sil = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_ens_all, cols)
#%%
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
df6_sil_cl = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_ens_all, 'H_Cluster')
df6_sil_cl_min= get_small_metric_row_ind(df6_sil_cl, error_metric)

#%%
padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_freq_All'
df6_freq = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_ens_all, cols)
#%%
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
df6_freq_cl = gen_df_row_dom(eval_dir_all,error_metric, padrao_dom_clu_euc_ens_all, 'H_Cluster')
df6_freq_cl_min= get_small_metric_row_ind(df6_freq_cl, error_metric)

#%%
### dominio with clusters dtw
padrao_dom_clu_dtw_all = 'H_dominio_cluster_dtw_All'
df7_ = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_all, cols)
#%%
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
df7_cl = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_all, 'H_Cluster')
df7_cl_min= get_small_metric_row_ind(df7_cl, error_metric)

padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_sil_All'
df8_sil = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_ens_all, cols)
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
df8_sil_cl = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_ens_all, 'H_Cluster')
df8_sil_cl_min= get_small_metric_row_ind(df8_sil_cl, error_metric)

padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_freq_All'
df8_freq = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_ens_all, cols)
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
df8_freq_cl = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_dtw_ens_all, 'H_Cluster')
df8_freq_cl_min= get_small_metric_row_ind(df8_freq_cl, error_metric)

#%%
'''
# ## Clusters euclidean -All
eval_file = [x for x in eval_files if 'H_cluster_euclidean_All' in x][0]
eval_H_cluster_euc = pd.read_pickle(eval_file)
df1_ = get_small_metric_row(eval_H_cluster_euc, error_metric) #H_Cluster13_All

eval_file = [x for x in eval_files if 'H_cluster_euclidean_ensemble_sil_All' in x][0]
eval_H_cluster_euc_ensemble = pd.read_pickle(eval_file)
#'''

#%%
# Clusters euclidean - All
padrao_clu = 'H_cluster_euclidean_All'
df1_ = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
# Clusters euclidean with ensemble - All
# ens silhouette
padrao_clu = 'H_cluster_euclidean_ensemble_sil_All'
df2_sil = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
#ens Frequency
padrao_clu = 'H_cluster_euclidean_ensemble_freq_All'
df2_freq = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)

#%%
### Clusters dtw

padrao_clu = 'H_cluster_dtw_All'
df3_ = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)       #['H_Cluster8_All']]
# Clusters dtw with ensemble - All
# ens silhouette
padrao_clu = 'H_cluster_dtw_ensemble_sil_All'
df4_sil = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)    #2.loc[['H_Cluster17_All']]
# ens Frequency
padrao_clu = 'H_cluster_dtw_ensemble_freq_All'
df4_freq = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)

#%%
result_eval_rmse_all = pd.concat([df1_, df2_sil, df2_freq, \
                             df3_, df4_sil, df4_freq, \
                             df5_, df5_cl_min, \
                             df6_sil, df6_sil_cl_min, df6_freq, df6_freq_cl_min,\
                             df7_, df7_cl_min, \
                             df8_sil, df8_sil_cl_min, df8_freq, df8_freq_cl_min,
                             df9_], axis=0) 
#del df1_,df2_, df3_,df4_, df5_,df6_, df7_,df8_, df9_

### 14/09/2023
def get_result_H_cluster_selection(eval_dir_all, error_metric, all='', seleuc='', seldtw=''):
    ''' not ready
    return df result 
    '''
    # Clusters euclidean - All
    if all:
        padrao_clu = 'H_cluster_euclidean_'+all
    elif seleuc:
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

#%%
result_eval_clusters_all = result_H_cluster_selection(eval_dir_all, error_metric, all='All', seleuc='', seldtw='')

### fim 14/09/2023

#%%
error_metric = 'rmse'
df1_ = get_small_metric_row(eval_H_cluster_euc, error_metric) #H_Cluster13_All
df2_ = get_small_metric_row(eval_H_cluster_euc_ensemble, error_metric) #[['H_Cluster2_All']]
#%%
eval_H_cluster_euc = df_cols_to_numeric(eval_H_cluster_euc)

mask1 = eval_H_cluster_euc.index.get_level_values('metric').isin(['rmse'])
eval_H_cluster_euc2 = eval_H_cluster_euc.loc[mask1]
#para excluir o bottom e pegar a hierarquia , pedo o segundo menor
index_second=eval_H_cluster_euc2['MinTrace(mint_shrink)'].nsmallest(2).index[1]
df1_= eval_H_cluster_euc2.loc[[index_second]]

#%%
df3_ = get_small_metric_row(eval_H_cluster_dtw, error_metric) #['H_Cluster8_All']]
df4_ = get_small_metric_row(eval_H_cluster_dtw_ensemble, error_metric)#2.loc[['H_Cluster17_All']]
#%%
#### Dominio
df5_ = get_rows_from_cols(eval_H_dom_cluster_euc, error_metric,'H_Dominio')
df6_ = get_rows_from_cols(eval_H_dom_cluster_euc_ensemble, error_metric,'H_Dominio') #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df7_= get_rows_from_cols(eval_H_dom_cluster_dtw, error_metric,'H_Dominio') #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df8_= get_rows_from_cols(eval_H_dom_cluster_dtw_ensemble, error_metric,'H_Dominio') #eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio']]
df9_= get_rows_from_cols(eval_H_dom, error_metric,'H_Dominio') #eval_H_dom2.loc[['H_Dominio']]

#%%
result_eval_rmse = pd.concat([df1_, df2_, df3_, df4_,df5_,df6_,df7_,df8_,df9_], axis=0) 
del df1_,df2_, df3_,df4_, df5_,df6_, df7_,df8_, df9_
##### end Eval for rmse

#%% ########## Eval for mase ##########
error_metric = 'mase'
# Clusters
#euc
df1_mase = get_small_metric_row(eval_H_cluster_euc, error_metric) #H_Cluster13_All
df2_mase = get_small_metric_row(eval_H_cluster_euc_ensemble, error_metric) #[['H_Cluster2_All']]

#dtw
df3_mase = get_small_metric_row(eval_H_cluster_dtw, error_metric) #['H_Cluster8_All']]
df4_mase = get_small_metric_row(eval_H_cluster_dtw_ensemble, error_metric)#2.loc[['H_Cluster17_All']]

#### Dominio
df5_mase = get_rows_from_cols(eval_H_dom_cluster_euc, error_metric,'H_Dominio')
df6_mase = get_rows_from_cols(eval_H_dom_cluster_euc_ensemble, error_metric,'H_Dominio') #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df7_mase = get_rows_from_cols(eval_H_dom_cluster_dtw, error_metric,'H_Dominio') #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df8_mase = get_rows_from_cols(eval_H_dom_cluster_dtw_ensemble, error_metric,'H_Dominio') #eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio']]
df9_mase = get_rows_from_cols(eval_H_dom, error_metric,'H_Dominio') #eval_H_dom2.loc[['H_Dominio']]

# concat all smalls
result_eval_mase = pd.concat([df1_mase, df2_mase, df3_mase, df4_mase,df5_mase,df6_mase,df7_mase,df8_mase,df9_mase], axis=0) 

#save result to file

eval_file='data/evals_tmp/evaluation_result_mase_H_dominio_cluster_20230816.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval_mase, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)


#%% ##eval for bottomss of all hierarchies
col = 'Bottom'
error_metric='mase'
df1 = get_rows_from_cols(eval_H_cluster_euc, error_metric,col)
df2 = get_rows_from_cols(eval_H_cluster_euc_ensemble, error_metric, col) #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df3 = get_rows_from_cols(eval_H_cluster_dtw, error_metric,col) #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df4 = get_rows_from_cols(eval_H_cluster_dtw_ensemble, error_metric,col)
df5 = get_rows_from_cols(eval_H_dom_cluster_euc, error_metric,col)
df6 = get_rows_from_cols(eval_H_dom_cluster_euc_ensemble, error_metric, col) #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df7 = get_rows_from_cols(eval_H_dom_cluster_dtw, error_metric,col) #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df8 = get_rows_from_cols(eval_H_dom_cluster_dtw_ensemble, error_metric,col) #eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio']]
df9 = get_rows_from_cols(eval_H_dom, error_metric,col) #eval_H_dom2.loc[['H_Dominio']]

result_eval_bottom_mase = pd.concat([df1, df2, df3, df4, df5,df6,df7,df8,df9], axis=0)
eval_file='data/evals_tmp/evaluation_result_bottom_mase_H_dominio_cluster_20230816.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval_mase, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)

#bottom rmse
error_metric='rmse'
df1 = get_rows_from_cols(eval_H_cluster_euc, error_metric,col)
df2 = get_rows_from_cols(eval_H_cluster_euc_ensemble, error_metric, col) #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df3 = get_rows_from_cols(eval_H_cluster_dtw, error_metric,col) #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df4 = get_rows_from_cols(eval_H_cluster_dtw_ensemble, error_metric,col)
df5 = get_rows_from_cols(eval_H_dom_cluster_euc, error_metric,col)
df6 = get_rows_from_cols(eval_H_dom_cluster_euc_ensemble, error_metric, col) #eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df7 = get_rows_from_cols(eval_H_dom_cluster_dtw, error_metric,col) #eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df8 = get_rows_from_cols(eval_H_dom_cluster_dtw_ensemble, error_metric,col) #eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio']]
df9 = get_rows_from_cols(eval_H_dom, error_metric,col) #eval_H_dom2.loc[['H_Dominio']]

result_eval_bottom_rmse = pd.concat([df1, df2, df3, df4, df5,df6,df7,df8,df9], axis=0)
eval_file='data/evals_tmp/evaluation_result_bottom_rmse_H_dominio_cluster_20230816.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval_mase, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)
#%%
#eval for HCluster individuals
#read all files for each strategy
def list_files_to_read( eval_dir, padrao):
    ''''
    get a list of files in a eval_dir based on a padrao
    '''
    if not eval_dir:
        eval_dir='data/evals_tmp/'


    #14/09/2023 padrao_ = os.path.join(eval_dir, '*_'+padrao+'.pkl') #euclidean_Individual.pkl')
    padrao_ = os.path.join(eval_dir, '*_'+padrao) #euclidean_Individual.pkl')

    print ("padrao: ", padrao_)

    eval_files = glob.glob(padrao_)
    print ("eval_files", eval_files)

    return eval_files
#%%
def read_files(eval_dir, padrao):
    ''''
    read a list of files based on a padrao
    '''

    eval_files = list_files_to_read(eval_dir, padrao)
    print (eval_files)

    eval_files_indiv = [x for x in eval_files if padrao in x] 
    eval_H_cluster_metric_ind={}

    for i,v in enumerate(eval_files_indiv):
        eval_H_cluster_metric_ind[i] = pd.read_pickle(eval_files_indiv[i])

    return eval_H_cluster_metric_ind

def eval_files_concat(eval_H_cluster_metric_ind):
    '''
    Concatenates a list of dataframes
    '''
    eval_H_cluster_metric_ind_full = pd.DataFrame()
    for i in eval_H_cluster_metric_ind.keys():
        if i==0:
            eval_H_cluster_metric_ind_full = eval_H_cluster_metric_ind[i]
            print (i)
            continue
        eval_H_cluster_metric_ind_full = pd.concat([eval_H_cluster_metric_ind_full,eval_H_cluster_metric_ind[i]], axis=0)

    return eval_H_cluster_metric_ind_full
#%%
def get_best_df_row_metric_H_cluster_ind(eval_dir, padrao, metric):
    '''
    returns the row with best metric value in df
    '''
    # 1. read files, concatenate them and get the row with min rmse value
    #eval_dir = 'data/evals_tmp/individuals'
    #padrao = 'euclidean_Individual'
    eval_H_cluster_ind = read_files(eval_dir, padrao)
    #concatenate all individual clusters df in only one df: 
    eval_H_cluster_ind_full = eval_files_concat(eval_H_cluster_ind)
    #
    #1.1 gets the rows with min rmse value
    df_min= get_small_metric_row_ind(eval_H_cluster_ind_full, metric , col_index_to_filter='level',bottom=0, both=0)

    return df_min
#%%
#### analysis for individual clusters
##### Euclidean

eval_dir = 'data/evals_tmp/individuals'
metric = 'rmse'

padrao = 'euclidean_Individual'
df_euc_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao, metric)
#%% 
############ Euclidean ensemble
padrao_ensemble = 'euclidean_ensemble_sil_Individual'
df_euc_ensemble_sil_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao_ensemble, metric)

padrao_ensemble = 'euclidean_ensemble_freq_Individual'
df_euc_ensemble_freq_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao_ensemble, metric)
############ fim euclidean
#%%
######### DTW
#eval_dir = 'data/evals_tmp/individuals'
padrao_dtw = 'dtw_Individual'
df_dtw_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao_dtw, metric)
#%%
######### DTW Ensemble
#eval_dir = 'data/evals_tmp/individuals/'
padrao_dtw_ens = 'dtw_ensemble_sil_Individual'
df_dtw_ensemble_sil_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao_dtw_ens, metric)

padrao_dtw_ens = 'dtw_ensemble_freq_Individual'
df_dtw_ensemble_freq_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao_dtw_ens, metric)
########### End DTW analysis
#%%
#### Adcionar o resultado destas hierarquias no result_table 
# e salvar num arquivo e gerar latex format
df_to_concat={}
df_to_concat[0] = result_eval_rmse_all 
df_to_concat[1] = df_euc_min
df_to_concat[2] = df_euc_ensemble_sil_min 
df_to_concat[3] = df_euc_ensemble_freq_min 
df_to_concat[4] = df_dtw_min 
df_to_concat[5] = df_dtw_ensemble_sil_min
df_to_concat[6] = df_dtw_ensemble_freq_min

result_eval_rmse_with_indiv= eval_files_concat(df_to_concat)

#%%
del df_to_concat
#%%
#save result to file
eval_file='data/evals_tmp/evaluation_result_rmse_H_dominio_cluster.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval_rmse_with_indiv, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)

#%% gere latex
#para savar a tabela no formato latex precisa retirar as colunas como indices, se index=False
result=result_eval_rmse_with_indiv.reset_index()
#%%
latex_table = result_eval_rmse_with_indiv.to_latex(index=True)
#%%
# Save the LaTeX code to a .tex file
latex_file_path = 'latex/tables/result_eval_rmse_with_indiv.tex'
with open(latex_file_path, 'w') as f:
    f.write(latex_table)
print(latex_table)
#%%
###Fazer a analise do Dominio com cadacluster individualmente
#### analysis for dominio with individual clusters
def set_index_value(df, ensemble):
    df_dom_ind = get_rows_from_cols(df, 'rmse','H_Dominio')
    ncluster=df.index.get_level_values('level')
    ncluster=[x for x in ncluster if 'H_Cluster' in x ]
    ncluster=ncluster[0].split('H_Cluster')[1]
    #ncluster=ncluster[0][-1]

    df_index = list(df_dom_ind.index.names)
    
    df_dom_ind.reset_index(inplace=True)
    if ensemble:
        df_dom_ind['Strategy']='Dominio Ind_Cluster'+ncluster+' Ensemble'
    else:
        df_dom_ind['Strategy']='Dominio Ind_Cluster'+ncluster
    df_dom_ind.set_index(df_index, inplace=True)

    return df_dom_ind
#%%
def get_best_dom_cluster_ind(eval_dir, metric,padrao, ensemble=0):

    eval_H_dom_cluster_ind = read_files(eval_dir, padrao)
    
    
    df_dom_ind_ens={}
    for n in eval_H_dom_cluster_ind.keys():
        df_dom_ind_ens[n] = set_index_value(eval_H_dom_cluster_ind[n],ensemble)
    
    #concatenate all individual clusters df in only one df: 
    eval_H_dom_cluster_ind_full = eval_files_concat(df_dom_ind_ens)
    
    #1.1 gets the rows with min rmse value
    df_dom_clu_min= get_small_metric_row(eval_H_dom_cluster_ind_full, metric )

    return df_dom_clu_min

#%%
##### Euclidean

eval_dir = 'data/evals_tmp/individuals/dominio_cluster'
metric = 'rmse'
#%%
padrao = 'euclidean_Individual'
df_dom_clu_euc_min = get_best_dom_cluster_ind(eval_dir, metric,padrao, 0)

#%%
padrao = 'euclidean_ensemble_Individual'

#df_dom_euc_min = get_best_df_row_metric_H_cluster_ind(eval_dir, padrao, metric)

#%%
#teste com funcao
df_dom_clu_euc_ens_min = get_best_dom_cluster_ind(eval_dir, metric,padrao, 1)

#%% 
##### dtw
padrao_dtw = 'dtw_Individual'
df_dom_clu_dtw_min = get_best_dom_cluster_ind(eval_dir, metric,padrao_dtw,0)
#%%
##### dtw ensemble
padrao_dtw = 'dtw_ensemble_Individual'
df_dom_clu_dtw_ens_min = get_best_dom_cluster_ind(eval_dir, metric,padrao_dtw,1)

#%%
#concatenate these results

#%%
eval_file='data/evals_tmp/evaluation_result_rmse_H_dominio_cluster.pkl'
#%%
#read the file with results
df_eval_= pd.read_pickle(eval_file) 
#or use 
#%%
#### Adcionar o resultado destas hierarquias no result_table 
# e salvar num arquivo e gerar latex format
df_to_concat={}
df_to_concat[0] = df_eval_ 
df_to_concat[1] = df_dom_clu_euc_min
df_to_concat[2] = df_dom_clu_euc_ens_min 
df_to_concat[3] = df_dom_clu_dtw_min 
df_to_concat[4] = df_dom_clu_dtw_ens_min
result_eval_rmse_with_indiv = eval_files_concat(df_to_concat)


#%%
#save result to file
eval_file_='data/evals_tmp/evaluation_result_rmse_H_dominio_cluster_All_ind.pkl'
with open(eval_file_,'wb') as handle:
    pickle.dump(result_eval_rmse_with_indiv, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)

#%% gere latex
#para savar a tabela no formato latex precisa retirar as colunas como indices, se index=False
result=result_eval_rmse_with_indiv.reset_index()
#%%
latex_table = result_eval_rmse_with_indiv.to_latex(index=True)
#%%
# Save the LaTeX code to a .tex file
latex_file_path = 'latex/tables/result_eval_rmse_with_all_indiv.tex'
with open(latex_file_path, 'w') as f:
    f.write(latex_table)
#%%
quit()
################################################ Para baixo sao testes
#%%
#2. get row with smallest rmse value for each cluster individual df
error_metric = 'rmse'

df_tmp={}
for x in eval_H_cluster_euc_ind:
    df_tmp[x]=pd.DataFrame()
    df_tmp[x] = get_small_metric_row_ind(eval_H_cluster_euc_ind[x], error_metric, col_index_to_filter='level',bottom=0, both=0)
#%%
df_tmp_full= eval_files_concat(df_tmp)



#%%
#gero um df com o menores de cada hierarquia e 
#%%
eval_H_cluster_euc_ind_hierarchies 
eval_H_cluster_euc_ind_hierarchies = get_small_metric_row(df_tmp, error_metric)
#%%
eval_dir = 'data/evals_tmp/'


# concatenate all and get the best(smaller) rmse
#%%


    #%%
quit()
#pegar todos os bottoms, e depois rmse e mase

#%%
#%%
# index_value_to_exclude = ('level','Bottom')
# mask2 = eval_H_cluster_euc.index != index_value_to_exclude
#eval_H_cluster_euc_2 = eval_H_cluster_euc_2.loc[mask2]
# eval_H_cluster_euc_2['MinTrace(mint_shrink)']=eval_H_cluster_euc_2['MinTrace(mint_shrink)'].astype(float)
# column_to_check = 'MinTrace(mint_shrink
# index_with_second_smallest_value = eval_H_cluster_euc_2[column_to_check].nsmallest(2).index[1]
# index_with_second_smallest_value

#%%
#%%
mask = eval_H_dom.index.get_level_values('metric').isin(['rmse'])
eval_H_dom2=eval_H_dom[mask]
#%%
mask1 = eval_H_dom_cluster_euc.index.get_level_values('metric').isin(['rmse'])
eval_H_dom_cluster_euc2 = eval_H_dom_cluster_euc.loc[mask1]
#%%
mask1 = eval_H_dom_cluster_euc_ensemble.index.get_level_values('metric').isin(['rmse'])
eval_H_dom_cluster_euc_ensemble2 = eval_H_dom_cluster_euc_ensemble.loc[mask1]
#%%
mask1 = eval_H_dom_cluster_dtw.index.get_level_values('metric').isin(['rmse'])
eval_H_dom_cluster_dtw2 = eval_H_dom_cluster_dtw.loc[mask1]
#%%
mask1 = eval_H_dom_cluster_dtw_ensemble.index.get_level_values('metric').isin(['rmse'])
eval_H_dom_cluster_dtw_ensemble2 = eval_H_dom_cluster_dtw_ensemble.loc[mask1]

#%%


mask1 = eval_H_cluster_dtw.index.get_level_values('metric').isin(['rmse'])
eval_H_cluster_dtw2 = eval_H_cluster_dtw.loc[mask1]

#%%
mask1 = eval_H_cluster_euc_ensemble.index.get_level_values('metric').isin(['rmse'])
eval_H_cluster_euc_ensemble2 = eval_H_cluster_euc_ensemble.loc[mask1]
#%%
mask1 = eval_H_cluster_dtw_ensemble.index.get_level_values('metric').isin(['rmse'])
eval_H_cluster_dtw_ensemble2 = eval_H_cluster_dtw_ensemble.loc[mask1]
#%%

eval_H_cluster_dtw_ensemble2['MinTrace(mint_shrink)']=pd.to_numeric(eval_H_cluster_dtw_ensemble2['MinTrace(mint_shrink)'])
#%%
index_second=eval_H_cluster_dtw_ensemble2['MinTrace(mint_shrink)'].nsmallest(2).index[1]

df4_=eval_H_cluster_dtw_ensemble2.loc[[index_second]]


#%%
df1=eval_H_cluster_euc2.loc[['H_Cluster13_All']]

#%%
df2=eval_H_cluster_euc_ensemble2.loc[['H_Cluster4_All']]

#%%
df3=eval_H_cluster_dtw2.loc[['H_Cluster8_All']]
#%%
df4=eval_H_cluster_dtw_ensemble2.loc[['H_Cluster17_All']]

#%%
df5=eval_H_dom_cluster_euc2.loc[['H_Dominio']]
df6=eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio']]
df7=eval_H_dom_cluster_dtw2.loc[['H_Dominio']]
df8=eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio']]
df9=eval_H_dom2.loc[['H_Dominio']]

#%%
result_eval = pd.concat([df1, df2, df3, df4,df5,df6,df7,df8,df9], axis=0) 
# %%
df10=df5=eval_H_dom_cluster_euc2.loc[['H_Dominio','H_Cluster6_All']]
# %%
df11=eval_H_dom_cluster_euc_ensemble2.loc[['H_Dominio','H_Cluster2_All']]

# %%
df12=eval_H_dom_cluster_dtw2.loc[['H_Dominio', 'H_Cluster13_All']]
df13=eval_H_dom_cluster_dtw_ensemble2.loc[['H_Dominio','H_Cluster3_All','H_Cluster16_All']]
# %%
result_eval2 = pd.concat([df1, df2, df3, df4,df10,df11,df12, df13, df9], axis=0) 
# %%


#%%
eval_file='../data/evals_tmp/evaluation_result_H_dominio_cluster_20230816.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval2, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)
# %%
