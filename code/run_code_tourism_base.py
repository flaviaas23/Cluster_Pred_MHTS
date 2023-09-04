import pandas as pd
import numpy as np
import os
import sys
import pickle

from preprocessing.preprocessing import Preprocessing 
from preprocessing.predict import Prediction
from hierarchicalforecast.evaluation import HierarchicalEvaluation

from utils.metricas import rmse, mase

def add_columns_info_to_eval_df(eval_df, strategy, d_metric, cluster_method):
    ''''
    Add columns to evaluation table to have same format with clustering
    '''
    eval_df = eval_df.reset_index()
    eval_df.insert(loc=1, column='Strategy', value= strategy)
    eval_df.insert(loc=3, column='DistMetric', value=dist_metric)
    eval_df.insert(loc=4, column='ClusterMethod', value=cluster_method)
    eval_df.set_index(['level', 'Strategy', 'metric','DistMetric'], inplace=True)
    eval_df = eval_df.sort_index(level=['level', 'Strategy', 'metric'])
    eval_df['SilAvg']=pd.NA

    return eval_df

if __name__=='__main__':

    print (os.getcwd())
    RAW_DIR = '../../Data_MHTS/'
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

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

    Y_df, S_df, tags = australia.aggregate_df(Y_df, spec)

    #print ("{}\n, {}\n", Y_df2.head(2), tags)

    ##### divide in train and test
    steps = 8
    Y_test_df, Y_train_df=australia.split_test_train(Y_df, steps)

    print (Y_test_df.shape, Y_train_df.shape, Y_df.shape)

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
    eval_tags['Purpose'] = tags['Country/Purpose']
    eval_tags['State'] = tags['Country/State']
    eval_tags['Regions'] = tags['Country/State/Region']
    eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    eval_tags['All'] = np.concatenate(list(tags.values()))
    eval_tags['H_Dominio'] = [*tags['Country'],\
                            *tags['Country/State'],\
                            *tags['Country/State/Region'],\
                            *tags['Country/State/Region/Purpose']]

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

    strategy_value = 'Dominio only'
    dist_metric = None #'dtw' #'euclidean'
    cluster_method = None

    evaluation2 = add_columns_info_to_eval_df(evaluation2, strategy_value, dist_metric, cluster_method)

    print (os.getcwd())
    eval_file_dom='data/evals_tmp/evaluation_H_dominio.pkl'
    with open(eval_file_dom,'wb') as handle:
        pickle.dump(evaluation2, handle, protocol=pickle.HIGHEST_PROTOCOL)


