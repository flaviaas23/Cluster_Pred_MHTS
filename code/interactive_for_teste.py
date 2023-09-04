
#%%
import pandas as pd
import numpy as np
import os
import sys

from preprocessing.preprocessing import Preprocessing 
from preprocessing.predict import Prediction
from cluster.clustering import Clustering

from hierarchicalforecast.evaluation import HierarchicalEvaluation

from utils.metricas import rmse, mase


print (os.getcwd())
RAW_DIR = '../../Data_MHTS/'
#RAW_FILE = 'tourism.csv'
RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
DATA_CLUSTER_DIR = 'data/cluster/'
pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
######### 1. Load and Process Data
australia = Preprocessing(RAW_DIR, RAW_FILE) 
Y_df=australia.load_preprocess_tourism()
spec = [
    ['Country'],
    ['Country', 'State'], 
    ['Country', 'Purpose'], 
    ['Country', 'State', 'Region'], 
    ['Country', 'State', 'Purpose'], 
    ['Country', 'State', 'Region', 'Purpose']
]
Y_df2, S_df, tags = australia.aggregate_df(Y_df, spec)

#load cluster   
cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
dic_cluster = cluster.load_cluster()

#add cluster info to dataframe and spec
Y_df_cluster, spec_cluster = cluster.gen_df_tourism_cluster(dic_cluster, Y_df)
print ("Ydfcluster:",spec_cluster, Y_df_cluster.shape,"\n", Y_df_cluster.head(2))

#%%
Y_df_cluster, S_df_cluster, tags_cluster = australia.aggregate_df(Y_df_cluster, spec_cluster)
    
print ("\nYdfcluster2:",tags_cluster, Y_df_cluster.shape,"\n", Y_df_cluster.head(2))
#print ("{}\n, {}\n", Y_df2.head(2), tags)

##### divide in train and test
steps = 8
Y_test_df, Y_train_df=australia.split_test_train(Y_df_cluster, steps)
print (Y_test_df.shape, Y_train_df.shape, Y_df_cluster.shape)
#######  2.Prediction
steps = 8
season_length = 4
model = 'ZZA'
freq = 'QS'
pred = Prediction()
Y_hat_df, Y_fitted_df= pred.predict_ets(Y_train_df, s_length = season_length, 
                                        md = model, fq = freq, h = steps
                                        )
print ("{}\n, {}\n".format(Y_hat_df.head(2), Y_fitted_df.head(2)))
####### 3. Reconcile forecasts
Y_rec_df= pred.rec_BU_MinTrace(Y_hat_df, Y_fitted_df, S_df, tags)
####### 4. Evaluation
eval_tags = {}
eval_tags['Total'] = tags['Country']
#eval_tags['Purpose'] = tags['Country/Purpose']
#eval_tags['State'] = tags['Country/State']
#eval_tags['Regions'] = tags['Country/State/Region']
#eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
#eval_tags['All'] = np.concatenate(list(tags.values()))
n_clusters = list(dic_cluster.keys())
for n in n_clusters:
    eval_tags['Cluster'+str(n)] = tags_
evaluator = HierarchicalEvaluation(evaluators=[rmse, mase])
evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
        tags=eval_tags, Y_df=Y_train_df
)
evaluation2 = evaluation.drop('Overall')
evaluation2.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']
evaluation2 = evaluation2.applymap('{:.2f}'.format)
print ("**** Evaluation Tourism Prediction ***")
print (evaluation2)


### teste for ensemble

#%%
import numpy as np
import pandas as pd

import os


#imports for pickle - read and save 
import pickle
#to follow processing
from tqdm import tqdm

#for getting date to save file
from datetime import datetime

#nao funciona no interactive window
# from cluster.clustering import Clustering

#%%
PROCESSED_DIR = 'data/processed/'

# arquivos resultados do clustering
SAVE_DIR = "data/cluster/"
DATA_CLUSTER_DIR = 'data/cluster/'

TMP_DIR = "../tmp/"
num_interval = 4  #interval to save temp results

#load cluster   
pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
# cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
# dic_cluster = cluster.load_cluster()
with open(DATA_CLUSTER_DIR+pickle_cluster_file, 'rb') as handle: 
            b = pickle.load(handle)
dic_cluster= {**b}
#%%
def cria_SimilarityMatrix(dic_cluster):
    '''
    Function to create a Frequency matrix based on TS clustering
    '''    
    n_clusters = list(dic_cluster.keys())[:3]    
    nrow = len(dic_cluster[n_clusters[0]]['cluster'][:5])
    print (nrow, len(n_clusters))
    s = (nrow, nrow)
    freq_matrix= np.zeros(s)

    for n in n_clusters:
        print ("n = ",n)
        sil = dic_cluster[n]['sample_silhouette_values'][:5]
        cluster = dic_cluster[n]['cluster'][:5]
        print ("sil= ",sil,"\ncluster = ",cluster)
        for i in range(0, (nrow)):            
            print ("i = ",i)
            for j in range(0, nrow):
                #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j])
                if cluster[i] == cluster[j]:
                     
                    freq = (sil[i]+sil[j]+2)/4
                    freq_matrix[i,j] += freq
                    #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j], freq)
        
        #print ("freq_matrix = \n", freq_matrix)
    freq_matrix= freq_matrix/len(n_clusters)
    print ("freq_matrix = \n", freq_matrix)
    return freq_matrix            
#                
             
                

# %%
#teste tabela evaluation

  

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C':[10,20]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})

# Concatenate along rows (axis=0)
result_row = pd.concat([df1, df2, df3], axis=0) #usar este

# Concatenate along columns (axis=1)
result_column = pd.concat([df1, df2], axis=1)
#%%
result_row =pd.concat([df1, df2], axis=0)
result_row = pd.concat([result_row, df3], axis=0)
# %%
# Create a sample DataFrame with a multi-index
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('B', 3)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 40], [50, 60]]
df = pd.DataFrame(data, index=index, columns=columns)
#%%
# Create a sample MultiIndex DataFrame
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('C', 3), ('D', 4)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 15], [50, 25], [70, 10]]
df = pd.DataFrame(data, index=index, columns=columns)

# List of index values to exclude
index_values_to_exclude = [('B', 2), ('C', 3)]

# Create a boolean mask for index values not in the exclusion list
mask = ~df.index.isin(index_values_to_exclude)

# Use the mask to retrieve rows with index values not in the exclusion list
filtered_df = df[mask]
#%%
def add_column_df(df, level_to_filter, label_to_filter, column_to_set, value):
    '''
    Add a column and value to a multiindex df  
    '''
     
    # Specify the level and label to filter rows based on the multi-index
    # level_to_filter = 'level'#'Index1'
    # label_to_filter = 'H_Cluster17' #'B'

    # # Column where you want to set the values
    # column_to_set = 'Sil_avg'

    # # New value to set in the specified column
    # value = 42
    if column_to_set not in df:
        df[column_to_set]=pd.NA
    
    # Use .xs() to retrieve the cross-section of the DataFrame
    cross_section = df.xs(key=label_to_filter, level=level_to_filter)
    
    # Update the specified column in the cross-section
    cross_section[column_to_set] = value
    
    # Update the original DataFrame with the modified cross-section
    df.loc[df.index.get_level_values(level_to_filter) == label_to_filter, :] = cross_section.values

    #return df

#%%
def add_cluster_info_to_eval_df(eval_df, strategy, d_metric, cluster_method):
    ''''
    Add columns cluster info to evaluation table
    '''
    eval_df = eval_df.reset_index()
    eval_df.insert(loc=1, column='Strategy', value= strategy)
    eval_df.insert(loc=3, column='DistMetric', value=d_metric)
    eval_df.insert(loc=4, column='ClusterMethod', value=cluster_method)
    #eval_df = eval_df.set_index('Strategy', append=True)
    #eval_df = eval_df.set_index('DistMetric', append=True)
    eval_df.set_index(['level', 'Strategy', 'metric','DistMetric'], inplace=True)
    eval_df = eval_df.sort_index(level=['level', 'Strategy', 'metric'])
    return eval_df
#%%
eval_file_cluster_dtw_ensemble='../data/evals_tmp/evaluation_H_cluster_dtw_ensemble_All.pkl'
eval_H_cluster_dtw_ensemble=pd.read_pickle(eval_file_cluster_dtw_ensemble)
#%%
eval_H_cluster_dtw_ensemble=add_cluster_info_to_eval_df(eval_H_cluster_dtw_ensemble,\
                                                        'All_cluster ensemble',\
                                                        'dtw', 'KMeans')

#%%
eval_file_cluster_euc_ensemble='../data/evals_tmp/evaluation_H_cluster_euclidean_ensemble_All.pkl'
eval_H_cluster_euc_ensemble=pd.read_pickle(eval_file_cluster_euc_ensemble)
#%%
eval_H_cluster_euc_ensemble=add_cluster_info_to_eval_df(eval_H_cluster_euc_ensemble,\
                                                        'All_cluster ensemble',\
                                                        'euclidean', 'KMeans')

#%%
df1=eval_H_cluster_euc_ensemble.loc[['H_Cluster16_All']]
df2=eval_H_cluster_dtw_ensemble.loc[['H_Cluster16_All']]
#%%
result_row=pd.concat([df1,df2], axis=0)
result_row=result_row.sort_index(level=['level', 'Strategy', 'metric'])#, 'DistMetric'])
#%%
#teste add a column estrategy to a multiindex df
eval_H_cluster_dtw_ensemble.insert(loc=0, column='Estrategy', value='All clusters Ensemble')
#%%
eval_H_cluster_dtw_ensemble = eval_H_cluster_dtw_ensemble.set_index('Estrategy', append=True)
#%%
#test add a index to a multindex df
# Create a sample DataFrame with a multi-index
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('C', 3)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 40], [50, 60]]
df = pd.DataFrame(data, index=index, columns=columns)

#%%
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('B', 3)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 40], [50, 60]]
df = pd.DataFrame(data, index=index, columns=columns)
#%%
# New index values for the added level
#new_index_values = ['X', 'Y', 'Z']

# Add a new index level to the DataFrame
df['NewIndex'] = 'All clusters ensembled'#new_index_values

# Set the new index level
df = df.set_index('NewIndex', append=True)


# %%
# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'Age': [25, 30, 22, 35, 28]}
df = pd.DataFrame(data)

# Define the condition to filter rows based on a specific column value
age_threshold = 30

# Create a boolean mask based on the condition
mask = df['Age'] >= age_threshold
#
# Apply the mask to the DataFrame to filter rows
filtered_df = df[mask]
# %%
# Create a sample MultiIndex DataFrame
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('C', 3)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 15], [50, 25]]
df = pd.DataFrame(data, index=index, columns=columns)

# Column to find the smallest value in
column_to_check = 'MinTrace(mint_shrink)'

# Find the index of the row with the smallest value in the specified column
index_with_smallest_value = df_cluster_euc_rmse[column_to_check].idxmin()

# Use .loc[] to retrieve the row with the smallest value
row_with_smallest_value = df_cluster_euc_rmse.loc[index_with_smallest_value]
# %%
row_with_smallest_value
# %%
index = pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('C', 3), ('D', 4)], names=['Index1', 'Index2'])
columns = ['Column1', 'Column2']
data = [[10, 20], [30, 40], [50, 60], [70, 80]]
df = pd.DataFrame(data, index=index, columns=columns)

# Index value to exclude
index_value_to_exclude = ('B', 2)

# Create a boolean mask for excluding the specified index value
mask = df.index != index_value_to_exclude

# Apply the mask to the MultiIndex DataFrame to exclude the row
filtered_df = df.loc[mask]

print(filtered_df)

# %%
####################################################################################
# #teste for pred for each cluster
import pandas as pd
import numpy as np
import os
import sys
import pickle

from preprocessing.preprocessing import Preprocessing 
from preprocessing.predict import Prediction
from cluster.clustering import Clustering

from hierarchicalforecast.evaluation import HierarchicalEvaluation

from utils.metricas import rmse, mase
#%%
print (os.getcwd())
RAW_DIR = '../../Data_MHTS/'
#RAW_FILE = 'tourism.csv'
RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
DATA_CLUSTER_DIR = 'data/cluster/'
#cluster pickle files without ensemble
#pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049.pkl' 
                    #'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
#pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046.pkl'
#                   #"Tourism_bottom_pivot_cluster_dtw_20230731_1718.pkl"
    
#file with cluster using dist matrix of ensembled clusters
pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_similarity_matrix_ensemble.pkl'
                      #"Tourism_bottom_pivot_cluster_euclidean_20230731_1706_similarity_matrix_ensemble.pkl'
#pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_similarity_matrix_ensemble.pkl'
                      #"Tourism_bottom_pivot_cluster_dtw_20230731_1718_similarity_matrix_ensemble.pkl"
######### 1. Load and Process Data
australia = Preprocessing(RAW_DIR, RAW_FILE) 
Y_df = australia.load_preprocess_tourism()
spec = [
    ['Country'],
    ['Country', 'State'], 
    ['Country', 'Purpose'], 
    ['Country', 'State', 'Region'], 
    ['Country', 'State', 'Purpose'], 
    ['Country', 'State', 'Region', 'Purpose']
]
Y_df2, S_df, tags = australia.aggregate_df(Y_df, spec) #precisa disso para gerar o Y_df com 
                                                       #a coluna de ids_unico
#load cluster   
cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
dic_cluster = cluster.load_cluster()



#%%
n_clusters = [2]  #list(dic_cluster.keys())
for n in n_clusters:
    dic_cluster_ind = {}
    dic_cluster_ind[n] = dic_cluster[n]
    #add cluster info to dataframe and spec
    #Y_df_cluster, spec_cluster = cluster.gen_df_tourism_cluster(dic_cluster, Y_df)
    Y_df_cluster, spec_cluster = cluster.gen_df_tourism_cluster(dic_cluster_ind, Y_df)
    print ("Ydfcluster:",spec_cluster, Y_df_cluster.shape,"\n", Y_df_cluster.head(2))
    #spec_cluster.append(['Country', 'State', 'Region', 'Purpose'])
    #for cluster hierarchical only add Country(Total) and bottom(sleaves)
    spec_cluster.insert(0,spec[0])
    spec_cluster.extend([spec[5]])
    print ("spec_cluster;\n",spec_cluster)
    Y_df_cluster, S_df_cluster, tags_cluster = australia.aggregate_df(Y_df_cluster, spec_cluster)
    
    print ("\nYdfcluster2:", Y_df_cluster.shape,"\n", Y_df_cluster.head(2))
    #print ("{}\n, {}\n", Y_df2.head(2), tags)
    #print ("tags: ", tags)
    #print ("tags_cluster: ", tags_cluster)
    #quit()
    ##### divide in train and test
    steps = 8
    Y_test_df, Y_train_df=australia.split_test_train(Y_df_cluster, steps)
    print (Y_test_df.shape, Y_train_df.shape, Y_df_cluster.shape)
    #######  2.Prediction
    steps = 8
    season_length = 4
    model = 'ZZA'
    freq = 'QS'
    pred = Prediction()
    Y_hat_df, Y_fitted_df= pred.predict_ets(Y_train_df, s_length = season_length, 
                                            md = model, fq = freq, h = steps
                                            )
    print ("{}\n, {}\n".format(Y_hat_df.head(2), Y_fitted_df.head(2)))
    ####### 3. Reconcile forecasts
    Y_rec_df= pred.rec_BU_MinTrace(Y_hat_df, Y_fitted_df, S_df_cluster, tags_cluster)
    ####### 4. Evaluation
    eval_tags = {}
    eval_tags['Total'] = tags['Country']
    # eval_tags['Purpose'] = tags['Country/Purpose']
    # eval_tags['State'] = tags['Country/State']
    # eval_tags['Regions'] = tags['Country/State/Region']
    eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    
    #n_clusters = list(dic_cluster.keys())
    n_clusters_ind = list(dic_cluster_ind.keys())
    for n in n_clusters_ind:
        eval_tags['Cluster'+str(n)+'_All'] = tags_cluster['Country/Cluster'+str(n)]
    
    # eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    # eval_tags['All'] = np.concatenate(list(tags.values()))
    # eval_tags['H_Dominio'] = [*tags['Country'],\
    #                 *tags['Country/State'],\
    #                 *tags['Country/State/Region'],\
    #                 *tags['Country/State/Region/Purpose']]
    eval_tags['All'] = np.concatenate(list(tags_cluster.values()))
    for n in n_clusters_ind:
        eval_tags['H_Cluster'+str(n)+'_All'] = [*tags['Country'],\
                        *tags_cluster['Country/Cluster'+str(n)],\
                        *tags['Country/State/Region/Purpose']
                ]
        
    evaluator = HierarchicalEvaluation(evaluators=[rmse, mase])
    evaluation = evaluator.evaluate(
            Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
            tags=eval_tags, Y_df=Y_train_df
    )
    evaluation2 = evaluation.drop('Overall')
    evaluation2.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']
    evaluation2 = evaluation2.applymap('{:.2f}'.format)
    print ("type evaluation", type(evaluation2))
    #adding silhouette average information to evaluation table
    for n in dic_cluster_ind.keys():
        evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(n), 'SilAvg', dic_cluster_ind[n]['silhouette_avg'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(n), 'SilAvg', dic_cluster_ind[n]['silhouette_avg'])
    type_h_pred_rec = 'Indiv' #each  if for prediction and reconcliation were used all clusters groups or
                    # each group in the hierarchy
    strategy_value = type_h_pred_rec+'_clusters' # Ensemble'
    dist_metric = 'euclidean' #'dtw' #'euclidean'
    cluster_method = 'KMeans'
    evaluation2 = cluster.add_cluster_info_to_eval_df(evaluation2,\
                                                    strategy_value,\
                                                    dist_metric, cluster_method)
    print ("**** Evaluation Cluster Prediction ***")
    print (evaluation2)
  print (os.getcwd())
