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
Gera as predicoes com os clusters do dataset tourism
teste
'''
if __name__=='__main__':

    print (os.getcwd())
    RAW_DIR = '../../../Data_MHTS/' #'../../../Data_MHTS/' mac puc '../Data_MHTS/' mac novo
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

    DATA_CLUSTER_DIR = 'data/cluster/'
    #1 cluster pickle files without ensemble
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049.pkl' 
                        #'Tourism_bottom_pivot_cluster_euclidean_20230731_1706.pkl'
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046.pkl'
    #                   #"Tourism_bottom_pivot_cluster_dtw_20230731_1718.pkl"
    
    #2 file with cluster using dist matrix of ensembled clusters
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_similarity_matrix_ensemble.pkl'
    #                   #"Tourism_bottom_pivot_cluster_euclidean_20230731_1706_similarity_matrix_ensemble.pkl"
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_similarity_matrix_ensemble.pkl'
                        #"Tourism_bottom_pivot_cluster_dtw_20230731_1718_similarity_matrix_ensemble.pkl"
    
    #3 file with cluster using dist matrix of ensembled clusters based on frequency of points together
    #pickle_cluster_file = 'Tourism_bottom_pivot_cluster_euclidean_20230814_0049_freq_similarity_matrix_ensemble.pkl'
    pickle_cluster_file = 'Tourism_bottom_pivot_cluster_dtw_20230814_0046_freq_similarity_matrix_ensemble.pkl'

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

    #para passar apenas alguns clusters 
    # dic_cluster17={}
    # dic_cluster17[17] = dic_cluster[17]
    #coloca num dic apenas um cluster por vez
    n_clusters = list(dic_cluster.keys())
    print ("n_clusters: ", n_clusters)
    for n in n_clusters:
        print ("n: ", n)
        dic_cluster_ind = {}
        dic_cluster_ind[n] = dic_cluster[n]

        #add cluster info to dataframe and spec
        Y_df_cluster, spec_cluster = cluster.gen_df_tourism_cluster(dic_cluster_ind, Y_df)

        #print ("Ydfcluster:",spec_cluster, Y_df_cluster.shape,"\n", Y_df_cluster.head(2))

        #spec_cluster.append(['Country', 'State', 'Region', 'Purpose'])
        spec_cluster.extend(spec)
        Y_df_cluster, S_df_cluster, tags_cluster = australia.aggregate_df(Y_df_cluster, spec_cluster)
        
        #print ("\nYdfcluster2:", Y_df_cluster.shape,"\n", Y_df_cluster.head(2))
        #print ("{}\n, {}\n", Y_df2.head(2), tags)
    
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
        eval_tags['Purpose'] = tags['Country/Purpose']
        eval_tags['State'] = tags['Country/State']
        eval_tags['Regions'] = tags['Country/State/Region']
        eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
        
        #n_clusters = list(dic_cluster.keys())
        n_clusters_ind = list(dic_cluster_ind.keys())
        for nn in n_clusters_ind:
            eval_tags['Cluster'+str(nn)] = tags_cluster['Country/Cluster'+str(nn)]
        
        # eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
        eval_tags['All'] = np.concatenate(list(tags.values()))
        eval_tags['H_Dominio'] = [*tags['Country'],\
                        *tags['Country/State'],\
                        *tags['Country/State/Region'],\
                        *tags['Country/State/Region/Purpose']]

        # eval tags for clusters hierarchies
        for nn in n_clusters_ind:
            eval_tags['H_Cluster'+str(nn)] = [*tags['Country'],\
                            *tags_cluster['Country/Cluster'+str(nn)],\
                            *tags['Country/State/Region/Purpose']
                    ]

        evaluator = HierarchicalEvaluation(evaluators=[rmse, mase])

        evaluation = evaluator.evaluate(
                Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
                tags=eval_tags, Y_df=Y_train_df
        )

        print ("evaluation\n", evaluation)
        evaluation2 = evaluation.drop('Overall')
        evaluation2.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']
        evaluation2 = evaluation2.applymap('{:.2f}'.format)

        #dic_cluster_ind has just one key
        #adding silhouette average, median and standard deviation information to evaluation table
        sil_values = dic_cluster_ind[n]['sample_silhouette_values']
        #media.append(np.mean(sil_values))
        mediana = np.median(sil_values)
        desvio = np.std(sil_values)
        for nn in dic_cluster_ind.keys(): 
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(nn), 'SilAvg', dic_cluster[nn]['silhouette_avg'])
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(nn), 'SilMedian', mediana)
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Cluster'+str(nn), 'SilStdDev', desvio)

            evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(nn), 'SilAvg', dic_cluster[nn]['silhouette_avg'])
            evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(nn), 'SilMedian', mediana)
            evaluation2 = cluster.add_column_df(evaluation2, 'level','Cluster'+str(nn), 'SilStdDev', desvio)
            
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Dominio', 'SilAvg', dic_cluster[nn]['silhouette_avg'])
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Dominio', 'SilMedian', mediana)
            evaluation2 = cluster.add_column_df(evaluation2, 'level','H_Dominio', 'SilStdDev', desvio)


        type_h_pred_rec = 'Individual' #each  if for prediction and reconcliation were used all clusters groups or
                        # each group in the hierarchy
        
        dist_metric = 'dtw' #'dtw' #'euclidean'
        cluster_method = 'KMeans'
        ens = ' Ens'        #'', ' Ens'
        type_ensemble = '_freq'     #_freq, _sil, ''
        strategy_value = 'Dom '+'Ind_'+str(n)+'_clusters'+ens+type_ensemble   # Ens'

        evaluation2 = cluster.add_cluster_info_to_eval_df(evaluation2,\
                                                        strategy_value,\
                                                        dist_metric, cluster_method)




        print ("**** Evaluation Tourism Prediction ***")
        print (evaluation2)

        print (os.getcwd())
        #eval_file='data/evals_tmp/evaluation_H_dominio_cluster_euclidean.pkl'
        #eval_file='data/evals_tmp/evaluation_H_dominio_cluster17_dtw.pkl'
        #eval_file='data/evals_tmp/evaluation_H_dominio_cluster_euclidean_ensemble.pkl'
        #eval_file='data/evals_tmp/evaluation_H_dominio_cluster_dtw_ensemble_All.pkl'
        #eval_file='data/evals_tmp/evaluation_H_dominio_cluster_{}_{}.pkl'.format(dist_metric, type_h_pred_rec)
        #eval_file='data/evals_tmp/individuals/dominio_cluster/evaluation_H_dominio_cluster_{}_{}_{}.pkl'.format(n,dist_metric, type_h_pred_rec)
        #eval_file='data/evals_tmp/individuals/dominio_cluster/evaluation_H_dominio_cluster_{}_{}_{}.pkl'.format(n, dist_metric, type_h_pred_rec)
        #eval_file='data/evals_tmp/individuals/dominio_cluster/evaluation_H_dominio_cluster_{}_{}_ensemble{}_{}.pkl'.format(n,dist_metric, type_ensemble, type_h_pred_rec)
        
        if not ens:
            eval_file='data/evals_tmp/individuals/dominio_cluster/evaluation_H_dominio_cluster_{}_{}_{}.pkl'.format(n, dist_metric, type_h_pred_rec)
        else: 
            eval_file='data/evals_tmp/individuals/dominio_cluster/evaluation_H_dominio_cluster_{}_{}_ensemble{}_{}.pkl'.format(n, dist_metric, type_ensemble, type_h_pred_rec)

        with open(eval_file,'wb') as handle:
            pickle.dump(evaluation2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print (eval_file) 