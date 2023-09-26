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
''''
Gera as predicoes apenas dos clusters do dataset tourism mas com todos os clusters 
nas predicoes e nas reconciliacoes
'''
if __name__=='__main__':

    print (os.getcwd())
    RAW_DIR = '../../../Data_MHTS/' # no apple novo ../../../Data_MHTS
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

    DATA_CLUSTER_DIR = 'data/cluster/'
    #1 cluster pickle files without ensemble
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049.pkl' 
                        #'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046.pkl'
    #                   #"Tourism_bottom_pivot_cluster_dtw_20230731_1718.pkl"
    
    #2 file with cluster using dist matrix of ensembled clusters based on silhouette
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_similarity_matrix_ensemble.pkl'
                          #"Tourism_bottom_pivot_cluster_euclidean_20230731_1706_similarity_matrix_ensemble.pkl"
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_similarity_matrix_ensemble.pkl'
                          #"Tourism_bottom_pivot_cluster_dtw_20230731_1718_similarity_matrix_ensemble.pkl"

    #3 file with cluster using dist matrix of ensembled clusters based on frequency of points together
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_freq_similarity_matrix_ensemble.pkl'
    pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_freq_similarity_matrix_ensemble.pkl'

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

    Y_df2, S_df, tags = australia.aggregate_df(Y_df, spec)

    #load cluster   
    cluster = Clustering(DATA_CLUSTER_DIR, pickle_cluster_file)
    dic_cluster_all = cluster.load_cluster()

    # sil thr=0.45 sem ensemble, 0,79 com ens sil, 0.79 com ens freq
    selected_clusters, stats_df = cluster.select_clusters(dic_cluster_all, sil_meas_thr=0.1 )

    print ("For {}, selected clusters= {}".format(pickle_cluster_file, selected_clusters))
    dic_cluster = {key: dic_cluster_all[key] for key in selected_clusters}

    len_sel_clusters = len(selected_clusters)
    if len_sel_clusters == len(dic_cluster_all):
        str_cluster = 'All'
    else:
        str_cluster = str(len_sel_clusters)      #+'_clusters'
    print ("str_cluster: ", str_cluster)

    #para passar apenas alguns clusters 
    # dic_cluster17={}
    # dic_cluster17[17] = dic_cluster[17]

    #add cluster info to dataframe and spec
    Y_df_cluster, spec_cluster = cluster.gen_df_tourism_cluster(dic_cluster, Y_df)

    #print ("Ydfcluster:",spec_cluster, Y_df_cluster.shape,"\n", Y_df_cluster.head(2))

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

    #print (Y_test_df.shape, Y_train_df.shape, Y_df_cluster.shape)

    #######  2.Prediction

    steps = 8
    season_length = 4
    model = 'ZZA'
    freq = 'QS'

    pred = Prediction()
    Y_hat_df, Y_fitted_df= pred.predict_ets(Y_train_df, s_length = season_length, 
                                            md = model, fq = freq, h = steps
                                            )

    #print ("{}\n, {}\n".format(Y_hat_df.head(2), Y_fitted_df.head(2)))

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
    n_clusters = list(dic_cluster.keys())
    for n in n_clusters:
        #eval_tags['Cluster'+str(n)+'_All'] = tags_cluster['Country/Cluster'+str(n)]
        #tirei o _All do level
        eval_tags['Cluster'+str(n)] = tags_cluster['Country/Cluster'+str(n)]
    
    # eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    # eval_tags['All'] = np.concatenate(list(tags.values()))
    # eval_tags['H_Dominio'] = [*tags['Country'],\
    #                 *tags['Country/State'],\
    #                 *tags['Country/State/Region'],\
    #                 *tags['Country/State/Region/Purpose']]

    eval_tags['All'] = np.concatenate(list(tags_cluster.values()))
    for n in n_clusters:
        #eval_tags['H_Cluster'+str(n)+'_All'] = [*tags['Country'],\
        eval_tags['H_Cluster'+str(n)] = [*tags['Country'],                                     
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
    for n in dic_cluster.keys():
        # evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(n)+'_All', 'SilAvg', dic_cluster[n]['silhouette_avg'])
        # evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(n)+'_All', 'SilAvg', dic_cluster[n]['silhouette_avg'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(n), 'SilAvg', dic_cluster[n]['silhouette_avg'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(n), 'SilMedian', stats_df.loc[n-2]['mediana'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(n), 'SilStdDev', stats_df.loc[n-2]['desvio'])

        evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(n), 'SilAvg', dic_cluster[n]['silhouette_avg'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(n), 'SilMedian', stats_df.loc[n-2]['mediana'])
        evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(n), 'SilStdDev', stats_df.loc[n-2]['desvio'])

    #set for fill table evaluation
    type_h_pred_rec = str_cluster #'All'  if for prediction and reconcliation were used all clusters groups or
                                  # part of clusters groups in the hierarchy
    
    dist_metric = 'dtw' #'dtw' #'euclidean'
    cluster_method = 'KMeans'
    ens = ' Ens'        #'', ' Ens'
    type_ensemble = '_freq'   #'_freq', '_sil', ''
    strategy_value = type_h_pred_rec+'_clusters'+ens+type_ensemble    # Ens sil' # '','Ens freq', 'Ens sil'

    evaluation2 = cluster.add_cluster_info_to_eval_df(evaluation2,\
                                                      strategy_value,\
                                                      dist_metric, cluster_method)
    print ("**** Evaluation Cluster Prediction ***")
    print (evaluation2)

    print (os.getcwd())
    #eval_file='data/evals_tmp/evaluation_H_dominio_cluster_euclidean.pkl'
    #eval_file='data/evals_tmp/evaluation_H_dominio_cluster17_dtw.pkl'
    #eval_file='data/evals_tmp/evaluation_H_cluster_euclidean_all.pkl'
    #eval_file='data/evals_tmp/evaluation_H_cluster_dtw_all.pkl'
    #eval_file='data/evals_tmp/evaluation_H_cluster_euclidean_ensemble_all.pkl'
    #eval_file='data/evals_tmp/evaluation_H_cluster_dtw_ensemble_all.pkl'
    if type_h_pred_rec == 'All':
        if not ens:
            eval_file='data/evals_tmp/All/evaluation_H_cluster_{}_{}clusters.pkl'.format(dist_metric, type_h_pred_rec)
        else:
            eval_file='data/evals_tmp/All/evaluation_H_cluster_{}_ensemble{}_{}clusters.pkl'.format(dist_metric,type_ensemble, type_h_pred_rec)
    else:
        if not ens:
            eval_file='data/evals_tmp/selected/evaluation_H_cluster_{}_{}clusters.pkl'.format(dist_metric, type_h_pred_rec)
        else:
            eval_file='data/evals_tmp/selected/evaluation_H_cluster_{}_ensemble{}_{}clusters.pkl'.format(dist_metric,type_ensemble, type_h_pred_rec)
    with open(eval_file,'wb') as handle:
        pickle.dump(evaluation2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print (eval_file)    