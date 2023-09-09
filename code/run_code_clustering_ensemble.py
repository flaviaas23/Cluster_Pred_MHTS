
#%%
import numpy as np
import pandas as pd

import os

# warnings.filterwarnings("ignore")

#imports for clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#import numpy

from tslearn.clustering import TimeSeriesKMeans


#imports for MDS
# from dtaidistance import dtw

#imports for pickle - read and save 
import pickle
#to follow processing
#from tqdm import tqdm

#for getting date to save file
#from datetime import datetime

from cluster.clustering import Clustering

import multiprocessing
nproc = multiprocessing.cpu_count()
'''
20230808: Program to create a frequency matrix(co-association) based on cluster (CALC_SIMILARITY = 1)
          Calculate the clusters using frequency matrix (CALC_SIMILARITY = 0)
'''

'''
# Criando processo para encontrar melhor qtde de clusters baseado na métrica silhouette
## funcao para calcular o cluster de dataset
# Entrada
# n_grupos: quantidade de clusters que deve ser identificado
# X_train: dataset que será classificado em de array
# algoritmo: que será usado para classificar
# metrica_distancia: que será usada com o algoritmo de classificacao
# seed: default = 997
#'''
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

#%%
PROCESSED_DIR = 'data/processed/'

# arquivos resultados do clustering
SAVE_DIR = "data/cluster/"
DATA_CLUSTER_DIR = 'data/cluster/'

#TMP_DIR = "../tmp/"
num_interval = 4  #interval to save temp results

#load cluster   
pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049.pkl'#'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
#pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046.pkl' #'Tourism_bottom_pivot_cluster_dtw_20230731_1718.pkl'
cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)

CALC_SIMILARITY = 1
type_similarity = "_freq"       # if similarity based on freq: "_freq", if based on sillhouette:''
if CALC_SIMILARITY:
    # #load cluster   
    # #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
    # pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230731_1718.pkl'
    #cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
    dic_cluster = cluster.load_cluster()
    print (dic_cluster.keys(), dic_cluster[2].keys())

    #calculate similarity matrix
    if not type_similarity:
        #based on silhouette value
        similarity_matrix = cluster.cria_SimilarityMatrix(dic_cluster)
    else:
        similarity_matrix = cluster.cria_SimilarityMatrix_freq(dic_cluster)
    print (similarity_matrix.shape)

    #save dic_cluster and its similarity_matrix to pickle 
    cluster.save_similarity_matrix(dic_cluster, similarity_matrix, type_similarity)
else: 
    # read similarity matrix from pickle file
    pickle_file_sim = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_similarity_matrix.pkl'#'Tourism_bottom_pivot_cluster_euclidean_20230731_1706_similarity_matrix.pkl'
    #
    # pickle_file_sim = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_similarity_matrix.pkl' #'Tourism_bottom_pivot_cluster_dtw_20230731_1718_similarity_matrix.pkl'
    dic_cluster_sim = cluster.load_similarity(DATA_CLUSTER_DIR, pickle_file_sim)

    dist_sim_matrix = 1 - dic_cluster_sim['similarity_matrix']

    n_clusters = list(dic_cluster_sim['dic_cluster'].keys())

    #name of the original pickle data file processed
    pickle_data_file_name = dic_cluster_sim['dic_cluster'][n_clusters[0]]['data_file_name']
    
    # calculate clusters using dist matrix based on similarity matrix   
    obj_cluster_sim = cluster.cria_obj_grupos_matrix(n_clusters, dist_sim_matrix,\
                                                     DATA_CLUSTER_DIR+pickle_file_sim,\
                                                     pickle_data_file_name)

    cluster.save_cluster_ensemble(obj_cluster_sim, pickle_file_sim)
quit()

#%%
#reading the tourism data
data_file= 'Tourism_bottom_pivot'
df_cluster_sample = pd.read_pickle(PROCESSED_DIR+data_file+'.pkl')
print (df_cluster_sample.head(2))
print ("shape: ",df_cluster_sample.shape)
start_cols_dias = 1
num_dias=df_cluster_sample.shape[1]
print (df_cluster_sample.head(2).iloc[:,start_cols_dias:num_dias])
#%%
#df_cluster_array=df_cluster_sample.iloc[:,6:].to_numpy()

#gera o array a apartir do sample do dataset com num_dias dias
#num_dias=df_cluster_sample.shape[1]-start_cols_dias   #1913
num_dias=df_cluster_sample.shape[1]

df_cluster_array_1k=df_cluster_sample.iloc[:,start_cols_dias:num_dias].to_numpy()
#num_dias=df_cluster_sample.shape[1]#
#print (df_cluster_array_1k)
print ("start_cols_dias {} , num_dias {}, shape array: {}".format(start_cols_dias, num_dias, df_cluster_array_1k.shape))

#print ("array:\n", df_cluster_array_1k)
#range_n_clusters = int(df_cluster_array_1k.shape[0]**0.5) #numero de clusters a ser encontrados
t_clusters = int(df_cluster_array_1k.shape[0]**0.5) #numero de clusters a ser encontrados
range_n_clusters = [ i for i in range(2, t_clusters+1) ]
print ("t_clusters, range_n_clusters",t_clusters, range_n_clusters)

#%%
obj_cluster = {}

#range_n_clusters =[2,3,4,5,6,7,8,9,10]
seed=997
distance_metric="dtw"     #"dtw"    #"euclidean"

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
        "data_file_name": PROCESSED_DIR+data_file+'.pkl',
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
# %%
