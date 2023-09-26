
#%%
import pandas as pd
import numpy as np
import os
import sys
import glob

import pickle
#%%
def df_cols_to_numeric(df):
    ''''
    Set type to be numeric of eval columns 
    '''
    for x in df.columns[1:]:
        df[x] = pd.to_numeric(df[x])
        #print (x)

    return df

def list_files_to_read( eval_dir, padrao):
    ''''
    get a list of files in a eval_dir based on a padrao
    '''
    if not eval_dir:
        eval_dir='data/evals_tmp/'


    #14/09/2023 padrao_ = os.path.join(eval_dir, '*_'+padrao+'.pkl') #euclidean_Individual.pkl')
    padrao_ = os.path.join(eval_dir, '*_'+padrao+'*') #euclidean_Individual.pkl')

    print ("list_files_to_read: padrao_: ", padrao_)

    eval_files = glob.glob(padrao_)
    print ("list_files_to_read: eval_files", eval_files)

    return eval_files

def read_files(eval_dir, padrao):
    ''''
    read a list of files based on a padrao
    '''

    eval_files = list_files_to_read(eval_dir, padrao)
    print ('read_files1: ', eval_files,"\npadrao: ", padrao)

    eval_files_indiv = [x for x in eval_files if padrao in x] 
    print ('read_files2: ', eval_files_indiv)
    eval_H_cluster_metric_ind={}

    for i,v in enumerate(eval_files_indiv):
        eval_H_cluster_metric_ind[i] = pd.read_pickle(eval_files_indiv[i])
    print ('read_files3: ', eval_H_cluster_metric_ind.keys())

    return eval_H_cluster_metric_ind

def get_rows_from_cols(df, e_metric, cols, col_index_to_filter='level'):
    ''''
    From df returns row filtered
    cols=columns index values to be filtered, for ex H_Dominio
    '''
    # print ('e_metric= , cols=, col_index_to_filter='.format(e_metric, cols, col_index_to_filter))
    # print (df.head(1))
    # filter the metric to be considered
    mask1 = df.index.get_level_values('metric').isin([e_metric])
    df2 = df.loc[mask1]
    #print (df2.head(1))
    #df2_ = df2.loc[[cols]]
    mask2 = df2.index.get_level_values(col_index_to_filter)
    mask2=[x for x in mask2 if cols in x]
    # print ('mask2',mask2)
    df2_ = df2.loc[mask2]
    # print (df2.head(1))
    return df2_
#%%
def gen_df_row_dom(eval_dir, error_metric, padrao, cols):
    '''
    gets row of error_metric of cols, for ex cols=H_Dominio, H_cluster
    '''
    eval_H = read_files(eval_dir, padrao)
    print ('gen_df_row_dom: ',eval_H)

    #pega a linha do df com o valor que está em col
    df = get_rows_from_cols(eval_H[0], error_metric, cols) #eval_H_dom2.loc[['H_Dominio']]

    return df 
#%%
def gen_df_row_cluster(eval_dir, error_metric, padrao):
    '''
    gets row with bets metric
    '''
    eval_H = read_files(eval_dir, padrao)
    #print (eval_H)

    df_clu={}
    for n in eval_H.keys():
        #df_clu[n] = get_rows_from_cols(eval_H[n], error_metric, 'H_Cluster')
        df_clu[n] = get_small_metric_row_ind(eval_H[n], error_metric)

    eval_H_cluster_full = eval_files_concat(df_clu)
    #print ("eval_H_cluster_full: \n", eval_H_cluster_full)

    # acho q as duas funcoes estao retornando a mesma coisa, vou usar a segunda mais completa
    #df = get_small_metric_row(eval_H[0], error_metric) #H_Cluster13_All
    #df2 = get_small_metric_row_ind(eval_H[0], error_metric, col_index_to_filter='level',bottom=0, both=0)
    df2 = get_small_metric_row_ind(eval_H_cluster_full, error_metric, col_index_to_filter='level', H_dom=1,bottom=0, both=0)
    #print ('gen_df_row_cluster df2: ', df2)
    #print ('gen_df_row_cluster df1: ', df)
    return eval_H_cluster_full#df2 
#%%
def get_small_metric_row(df, e_metric, both=0):
    ''''
    Receives a df and returns its row with small column metric value
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
def get_small_metric_row_ind(df, e_metric, col_index_to_filter='level', H_dom=0, bottom=0, both=0):
    ''''
    20230817:Criei esta funcao para fazer um filtro do level para pegar os
    indices que tem Cluster numa parte deles
    depois pensar numa forma de fazer junto com a funcao acima
    Receives a df and returns its row with small column metric value
    se a coluna de level tem valores iguais nao deve ser feito o filtro do mask2
    '''
    #makes sure the errors columns values are numeric
    df = df_cols_to_numeric(df)

    # filter the metric to be considered
    
    mask1 = df.index.get_level_values('metric').isin([e_metric])
    df2 = df.loc[mask1]
    
    if not H_dom:
        #to get all indexes
        mask2 = df2.index.get_level_values(col_index_to_filter)    
        
        if bottom:
            #if want to exclude just bottom
            mask2=[x for x in mask2 if "Bottom" not in x ]
            print (mask2)
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
def get_result_H_dom_cluster_selection(eval_dir, error_metric, all='', seleuc='', seldtw=''):
    ''''
    return result of analisys with dominio clusters selection
    '''
    print ('all={}, seleuc={}, seldtw={}'.format(all, seleuc, seldtw))
    if all:
        eval_dir=eval_dir+all
        padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_'+all
        padrao_dom_clu_dtw_all = 'H_dominio_cluster_dtw_'+all
    elif seleuc:
        eval_dir=eval_dir+'selected/'
        padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_'+seleuc
    ### 1 dominio with clusters euclidean
    
    #padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_All'
    df5_ = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_euc_all, cols='H_Dominio')

    #verificar qual Hierarquia de cluster tem menor rmse na estrategia dom+cluster
    df5_cl = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_euc_all, 'H_Cluster')
    df5_cl_min= get_small_metric_row_ind(df5_cl, error_metric)

    ### 1.1 dominio with clusters EUCLIDEAN ENSEMBLE SILHOUETTE
    if all:
        padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_sil_'+all
    elif seleuc:
        padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_sil_'
    
    #padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_sil_All'
    df6_sil = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_euc_ens_all, cols)

    #verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
    df6_sil_cl = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_euc_ens_all, 'H_Cluster')
    df6_sil_cl_min= get_small_metric_row_ind(df6_sil_cl, error_metric)

    ### 1.2 dominio with clusters EUCLIDEAN ENSEMBLE FREQUENCY
    if all:
        padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_freq_'+all
    elif seleuc:
        padrao_dom_clu_euc_ens_all = 'H_dominio_cluster_euclidean_ensemble_freq_'
    df6_freq = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_euc_ens_all, cols)
    
    #verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
    df6_freq_cl = gen_df_row_dom(eval_dir,error_metric, padrao_dom_clu_euc_ens_all, 'H_Cluster')
    df6_freq_cl_min= get_small_metric_row_ind(df6_freq_cl, error_metric)

    ####### dominio with clusters dtw #######
    if seldtw:
        #eval_dir=eval_dir+'selected/'
        padrao_dom_clu_dtw_all = 'H_dominio_cluster_dtw_'+seldtw

    #padrao_dom_clu_dtw_all = 'H_dominio_cluster_dtw_All'
    df7_ = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_all, cols)
    
    #verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
    df7_cl = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_all, 'H_Cluster')
    df7_cl_min= get_small_metric_row_ind(df7_cl, error_metric)

    if all:
        padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_sil_'+all
    elif seldtw:
        padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_sil_'

    df8_sil = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_ens_all, cols)

    #verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
    df8_sil_cl = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_ens_all, 'H_Cluster')
    df8_sil_cl_min= get_small_metric_row_ind(df8_sil_cl, error_metric)

    if all:
        padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_freq_'+all
    elif seldtw:
        padrao_dom_clu_dtw_ens_all = 'H_dominio_cluster_dtw_ensemble_freq_'
    df8_freq = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_ens_all, cols)
    
    #verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom + cluster
    df8_freq_cl = gen_df_row_dom(eval_dir, error_metric, padrao_dom_clu_dtw_ens_all, 'H_Cluster')
    df8_freq_cl_min= get_small_metric_row_ind(df8_freq_cl, error_metric)

    result_eval = pd.concat([df5_, df5_cl_min, \
                             df6_sil, df6_sil_cl_min, df6_freq, df6_freq_cl_min,\
                             df7_, df7_cl_min, \
                             df8_sil, df8_sil_cl_min, df8_freq, df8_freq_cl_min
                             ], axis=0) 
    
    return result_eval


#%%
#fucntions for individuals
#%%
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

def get_best_dom_cluster_ind(eval_dir, metric, padrao, hierarq):
    '''
    hierarq= 'H_Dominio' ou H_cluster
    '''

    eval_H_dom_cluster_ind = read_files(eval_dir, padrao)
    
    
    # df_dom_ind_ens={}
    # for n in eval_H_dom_cluster_ind.keys():
    #     df_dom_ind_ens[n] = set_index_value(eval_H_dom_cluster_ind[n],ensemble)
    
    #concatenate all individual clusters df in only one df: 
    #eval_H_dom_cluster_ind_full = eval_files_concat(df_dom_ind_ens)

    df_dom_ind={}
    for n in eval_H_dom_cluster_ind.keys():
        df_dom_ind[n] = get_rows_from_cols(eval_H_dom_cluster_ind[n], metric, hierarq)

    eval_H_dom_cluster_ind_full = eval_files_concat(df_dom_ind)

    
    #1.1 gets the rows with min rmse value
    df_dom_clu_min= get_small_metric_row_ind(eval_H_dom_cluster_ind_full, metric, H_dom=1 )

    return df_dom_clu_min
#%%
def get_result_H_dom_cluster_individuals(eval_dir, error_metric):
    '''
    ok return df result od dom + cluster individually
    '''    
    
    eval_dir_ind = eval_dir+'individuals/dominio_cluster'

    padrao = 'euclidean_Individual'
    df_dom_euc_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Dominio')
    df_clu_euc_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Cluster')

    padrao = 'euclidean_ensemble_sil'
    df_dom_euc_ens_sil_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Dominio')
    df_clu_euc_ens_sil_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Cluster')

    padrao = 'euclidean_ensemble_freq'
    df_dom_euc_ens_freq_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Dominio')
    df_clu_euc_ens_freq_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Cluster')

    padrao = 'dtw_Individual'
    df_dom_dtw_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric,padrao, hierarq='H_Dominio')
    df_clu_dtw_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric,padrao, hierarq='H_Cluster')

    padrao = 'dtw_ensemble_sil'
    df_dom_dtw_ens_sil_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Dominio')
    df_clu_dtw_ens_sil_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Cluster')

    padrao = 'dtw_ensemble_freq'
    df_dom_dtw_ens_freq_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Dominio')
    df_clu_dtw_ens_freq_min = get_best_dom_cluster_ind(eval_dir_ind, error_metric, padrao, hierarq='H_Cluster')

    #### Adcionar o resultado destas hierarquias no result_table 
    # e salvar num arquivo e gerar latex format
    df_to_concat={}
     
    df_to_concat[0] = df_dom_euc_min
    df_to_concat[1] = df_clu_euc_min
    df_to_concat[2] = df_dom_euc_ens_sil_min 
    df_to_concat[3] = df_clu_euc_ens_sil_min 
    df_to_concat[4] = df_dom_euc_ens_freq_min
    df_to_concat[5] = df_clu_euc_ens_freq_min

    df_to_concat[6] = df_dom_dtw_min
    df_to_concat[7] = df_clu_dtw_min
    df_to_concat[8] = df_dom_dtw_ens_sil_min 
    df_to_concat[9] = df_clu_dtw_ens_sil_min 
    df_to_concat[10] = df_dom_dtw_ens_freq_min
    df_to_concat[11] = df_clu_dtw_ens_freq_min
    result_eval_indiv = eval_files_concat(df_to_concat)

    return result_eval_indiv

#%%
#fucntions for Clusters only 
def get_result_H_cluster_selection(eval_dir_all, error_metric, all='', seleuc='', seldtw=''):
    '''
    return df result for all cluster or a selection of clusters
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
#functions for clusters individuals analysis
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
def get_result_H_cluster_individuals(eval_dir_ind, error_metric):
    '''
    return df with best metric of each strategy for only clusters
    '''
    padrao = 'euclidean_Individual'
    df_euc_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao, error_metric)

    padrao_ensemble = 'euclidean_ensemble_sil_Individual'
    df_euc_ensemble_sil_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_ensemble, error_metric)

    padrao_ensemble = 'euclidean_ensemble_freq_Individual'
    df_euc_ensemble_freq_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_ensemble, error_metric)

    padrao_dtw = 'dtw_Individual'
    df_dtw_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_dtw, error_metric)

    padrao_dtw_ens = 'dtw_ensemble_sil_Individual'
    df_dtw_ensemble_sil_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_dtw_ens, error_metric)

    padrao_dtw_ens = 'dtw_ensemble_freq_Individual'
    df_dtw_ensemble_freq_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_dtw_ens, error_metric)

    df_to_concat={}
    df_to_concat[0] = df_euc_min
    df_to_concat[1] = df_euc_ensemble_sil_min 
    df_to_concat[2] = df_euc_ensemble_freq_min 
    df_to_concat[3] = df_dtw_min 
    df_to_concat[4] = df_dtw_ensemble_sil_min
    df_to_concat[5] = df_dtw_ensemble_freq_min

    result_eval= eval_files_concat(df_to_concat)
    return result_eval

#%%
### Dominio only
error_metric= 'rmse'
padrao = 'H_dominio.'
cols = 'H_Dominio'
eval_dir='data/evals_tmp/'

#get just the dominio
df9_ = gen_df_row_dom(eval_dir, error_metric, padrao, cols)

#dominio + clusters All
df_result_dom_cluster_all = get_result_H_dom_cluster_selection(eval_dir, error_metric,
                                                               all='All')

#dominio + clusters selected
df_result_dom_cluster_selection = get_result_H_dom_cluster_selection(eval_dir, 
                                        error_metric, seleuc='6', seldtw='8')

#%%
#dominio + clusters individually
eval_dir_ind = eval_dir+'individuals/dominio_cluster'
df_result_dom_cluster_individuals = get_result_H_dom_cluster_individuals(eval_dir, error_metric)
#%%
#clusters All
eval_dir_all = eval_dir+'All/'
df_result_cluster_all = get_result_H_cluster_selection(eval_dir_all, error_metric, all='All')
#%%
#clusters selection
eval_dir_sel = eval_dir+'selected/'
df_result_cluster_sel = get_result_H_cluster_selection(eval_dir_sel, error_metric, seleuc='6', seldtw='8')

#%%
#clusters individuals
eval_dir_ind = eval_dir+'individuals/'
df_result_cluster_individuals = get_result_H_cluster_individuals(eval_dir_ind, error_metric)

#%% 
############ Euclidean ensemble
padrao_ensemble = 'euclidean_ensemble_sil_Individual'
df_euc_ensemble_sil_min = get_best_df_row_metric_H_cluster_ind(eval_dir_ind, padrao_ensemble, error_metric)

#%%
# concatenate all results
result_eval_rmse_all = pd.concat([df9_, 
                             df_result_dom_cluster_all.sort_values(by='level', ascending=False),
                             df_result_dom_cluster_selection.sort_values(by='level', ascending=False),
                             df_result_dom_cluster_individuals.sort_values(by='level', ascending=False),
                             df_result_cluster_all.sort_values(by='level', ascending=False),
                             df_result_cluster_sel.sort_values(by='level', ascending=False),
                             df_result_cluster_individuals.sort_values(by='level', ascending=False) 
                             ], axis=0) 

#%%
#
#save result to file
eval_file='data/evals_tmp/evaluation_result_rmse_all.pkl'
with open(eval_file,'wb') as handle:
    pickle.dump(result_eval_rmse_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
print (eval_file)
#%%
latex_table = result_eval_rmse_all.to_latex(index=True)
#%%
# Save the LaTeX code to a .tex file
latex_file_path = 'latex/tables/result_eval_rmse_all.tex'
with open(latex_file_path, 'w') as f:
    f.write(latex_table)


############################################################################################################
############################################################################################################
#%%
### dominio with clusters euclidean
eval_dir_all=eval_dir+'All/'
padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_All'
df5_ = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_all, cols)

#%%
#verificar qual Hierarquia de cluster tem menor  rmse na estrategia dom+cluster
df5_cl = gen_df_row_dom(eval_dir_all, error_metric, padrao_dom_clu_euc_all, 'H_Cluster')
df5_cl_min= get_small_metric_row_ind(df5_cl, error_metric)

#%%

eval_dir_sel=eval_dir+'selected/'
padrao_dom_clu_euc_sel = 'H_dominio_cluster_euclidean_6'
dfs_ = gen_df_row_dom(eval_dir_sel, error_metric, padrao_dom_clu_euc_sel, cols)

#%%
padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_ensemble_sil_9'
dfs_sil9 = gen_df_row_dom(eval_dir_sel, error_metric, padrao_dom_clu_euc_all, cols)

#%%
padrao_dom_clu_euc_all = 'H_dominio_cluster_euclidean_ensemble_freq'
dfs_freq = gen_df_row_dom(eval_dir_sel, error_metric, padrao_dom_clu_euc_all, cols)
#%%
padrao_clu = 'H_cluster_euclidean_6'
df1_ = gen_df_row_cluster(eval_dir_sel, error_metric, padrao_clu)
#%%
padrao_clu = 'H_cluster_euclidean_ensemble_sil'
df2_sil = gen_df_row_cluster(eval_dir_sel, error_metric, padrao_clu)
#%%
padrao_clu = 'H_cluster_euclidean_ensemble_freq'
df2_freq = gen_df_row_cluster(eval_dir_sel, error_metric, padrao_clu)
#%%
padrao_clu = 'H_cluster_dtw_All'
df3_ = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)       #['H_Cluster8_All']]
#%%
# Clusters dtw with ensemble - All
# ens silhouette
padrao_clu = 'H_cluster_dtw_ensemble_sil_All'
df4_sil = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)    #2.loc[['H_Cluster17_All']]
#%%
# ens Frequency
padrao_clu = 'H_cluster_dtw_ensemble_freq_All'
df4_freq = gen_df_row_cluster(eval_dir_all, error_metric, padrao_clu)
#%%
result_eval_rmse_all = pd.concat([df1_, df2_sil, df2_freq, \
                             df3_, df4_sil, df4_freq, \
                             ], axis=0) 
#