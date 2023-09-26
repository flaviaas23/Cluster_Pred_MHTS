
import numpy as np
import pandas as pd

import os

# warnings.filterwarnings("ignore")

#imports for clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#import numpy
# import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
# from tslearn.datasets import CachedDatasets
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
#     TimeSeriesResampler

#imports for MDS
# from dtaidistance import dtw

# from dtaidistance import dtw_visualisation as dtwvis
# import random

# from sklearn.manifold import MDS
# from scipy.spatial.distance import pdist,  squareform

#imports for pickle - save the result to a file
import pickle
#to follow processing
from tqdm import tqdm

#for getting date to save file
from datetime import datetime

import multiprocessing
nproc = multiprocessing.cpu_count()-2
'''
Gera os clusters para o dataset tourism
'''

# Criando processo para encontrar melhor qtde de clusters baseado na métrica silhouette
## funcao para calcular o cluster de dataset
# Entrada
# n_grupos: quantidade de clusters que deve ser identificado
# X_train: dataset que será classificado em de array
# algoritmo: que será usado para classificar
# metrica_distancia: que será usada com o algoritmo de classificacao
# seed: default = 997
def cria_grupos(n_grupos, X_train, algoritmo, metrica_distancia, seed=997):
    if algoritmo == "Kmeans":
        if metrica_distancia == "dtw":
            dba_km = TimeSeriesKMeans(n_clusters=n_grupos,
                                      n_init=2,
                                      metric=metrica_distancia,
                                      verbose=True,
                                      max_iter_barycenter=10,
                                      random_state=seed,
                                      n_jobs=nproc)

        elif metrica_distancia == "euclidean":
            dba_km = TimeSeriesKMeans(n_clusters=n_grupos,
                                      n_init=2,
                                      metric="euclidean",
                                      verbose=True,
                                      max_iter=100,
                                      random_state=seed,
                                      n_jobs=nproc)

        dba_km_fit = dba_km.fit(X_train) 
        clusters_centers = dba_km_fit.cluster_centers_
        cluster_labels = dba_km_fit.labels_
        #cluster_labels = dba_km.fit_predict(X_train)


    return cluster_labels, clusters_centers


PROCESSED_DIR = 'data/processed/'

# arquivos resultados do clustering
#SAVE_DIR = "data/cluster/"

#TMP_DIR = "../tmp/"
num_interval = 4  #interval to save temp results

# calendar= pd.read_csv(INPUT_DIR+"/calendar.csv")
# sales_train_val = pd.read_csv(INPUT_DIR+"/sales_train_validation.csv")
# sell_prices = pd.read_csv(INPUT_DIR+"/sell_prices.csv")

'''  fas20230731
#Clustering
LOAD_DATA="pickle"
if LOAD_DATA == "pickle":
    with open(INPUT_DIR+'/pickle_sales_train_validation_price', 'rb') as handle:
        b = pickle.load(handle)

    print ("data file: ", b.keys())

    df_cluster = b['sales_train_validation_price']
else:
    df_cluster =sales_train_val.copy()
    df_cluster['id'] = df_cluster.apply(lambda row: row['id'].split('_validation')[0],axis=1)

print (df_cluster.shape)

group_dept = True
if group_dept:
    #grouping the series 
    df_70 = df_cluster.groupby(['store_id', 'dept_id']).sum().reset_index()
    df_70.insert(0,'dept_store_id', df_70['dept_id']+'_'+df_70['store_id'])
    num_series=df_70.shape[0]
    df_cluster_sample = df_70
    start_cols_dias = 3
else:
    #sampling
    #random fazendo groupby por estado, selecionando 20 series de cada categoria
    num_series=df_cluster.shape[0]  #30000
    if num_series < 30000:
        df_cluster_sample=df_cluster.sample(n=num_series,random_state=1).sort_index()
    else:
        df_cluster_sample = df_cluster
    
    start_cols_dias = 6
#fas 20230731 
''' 

gefcom=True
if gefcom:
    #read the preprocessed file
    data_file= 'gefcom2017_Y_df_bottom_pivot_df_cluster_sample'
    file_to_read = PROCESSED_DIR+'gefcom2017/'+data_file+'.pkl'
    SAVE_DIR = "data/cluster/gefcom2017/"

else:
    #reading the tourism data
    data_file= 'Tourism_bottom_pivot'
    file_to_read = PROCESSED_DIR+data_file+'.pkl'
    SAVE_DIR = "data/cluster/"

df_cluster_sample = pd.read_pickle(file_to_read)
#print (df_cluster_sample.head(2))
print ("shape: ",df_cluster_sample.shape)
start_cols_dias = 1
num_dias=df_cluster_sample.shape[1]
#print (df_cluster_sample.head(2).iloc[:,start_cols_dias:num_dias])

#df_cluster_array=df_cluster_sample.iloc[:,6:].to_numpy()

#gera o array a apartir do sample do dataset com num_dias dias
#num_dias=df_cluster_sample.shape[1]-start_cols_dias   #1913
num_dias = df_cluster_sample.shape[1]

df_cluster_array_1k = df_cluster_sample.iloc[:,start_cols_dias:num_dias].to_numpy()
#num_dias=df_cluster_sample.shape[1]#
#print (df_cluster_array_1k)
print ("start_cols_dias {} , num_dias {}, shape array: {}".format(start_cols_dias, num_dias, df_cluster_array_1k.shape))

#print ("array:\n", df_cluster_array_1k)
#range_n_clusters = int(df_cluster_array_1k.shape[0]**0.5) #numero de clusters a ser encontrados
t_clusters = int(df_cluster_array_1k.shape[0]**0.5) #numero de clusters a ser encontrados
range_n_clusters = [ i for i in range(2, t_clusters+1) ]
print ("t_clusters, range_n_clusters",t_clusters, range_n_clusters)

#
obj_cluster = {}

#range_n_clusters =[2,3,4,5,6,7,8,9,10]
seed=997
distance_metric="euclidean"     #"dtw"    #"euclidean"

#
for n_clusters in tqdm(range(2, t_clusters+1)):
    #print ("n_clusters= ", n_clusters)
    #cluster_labels[n_cluster],silhouette_avg[n_clusters],sample_silhouette_values[n_clusters]=cria_grupos(n_clusters, df_cluster_array_1k, "Kmeans", "dtw", seed=997)
    clusters, clusters_center = cria_grupos(n_clusters, df_cluster_array_1k, "Kmeans", distance_metric, seed=997)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df_cluster_array_1k, clusters)
    #print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :",silhouette_avg,
    #    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df_cluster_array_1k, clusters)
    #print ("silhouette_avg",silhouette_avg)

    obj_cluster[n_clusters] = {
        # "data": df_cluster_sample,# amostra
        "data_file_name": file_to_read, # PROCESSED_DIR+data_file+'.pkl',
        "seed": seed,
        "distance_metric": distance_metric,
        "cluster": clusters,     # [] resultado do cluster na amostra
        "clusters_centers": clusters_center,
        "dias_sample": num_dias,     #dias usados do sample
        "silhouette_avg":silhouette_avg, # silhouette_avg
        "sample_silhouette_values":sample_silhouette_values,        #[] resultado do silhoute para cada ponto do cluster
    }
    #fas 20233107 nao vou salvar mais os dados, só o nome, comentei abaixo
    #to save temps files , fas 20230731 removi para testar o prog primeiro
    # if not (n_clusters%num_interval) and n_clusters!=0:
    #     pickle_file_tmp = "pickle_tmp_" + str(n_clusters) + "_" + datetime.today().strftime('%Y%m%d_%H%M')
    #     #print("salvo i=", n_clusters, TMP_DIR +pickle_file_tmp)
    #     with open(TMP_DIR + pickle_file_tmp, 'wb') as handle:
    #         pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)


#para criar e salvar o arquivo com os dados do cluster 
pickle_cluster_file = data_file+"_cluster_"+distance_metric+"_"+datetime.today().strftime('%Y%m%d_%H%M')

with open(SAVE_DIR+pickle_cluster_file+'.pkl', 'wb') as handle:
    pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

#fas 20233107 nao vou salvar mais os dados, só o nome, comentei abaixo
# para criar e salvar o arquivo com os dados da amostra, 
# pickle_data_file = "pickle_data_"+distance_metric+"_"+str(num_series)+"_"+datetime.today().strftime('%Y%m%d_%H%M')
# with open(SAVE_DIR+pickle_data_file, 'wb') as handle:
#     pickle.dump(df_cluster_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print ("arquivo com os  dados: ", pickle_data_file)
print ("arquivo com o cluster: ", pickle_cluster_file)
# 
