#%%
import pandas as pd
import numpy as np
import os
import sys

from preprocessing.preprocessing import Preprocessing 
from preprocessing.predict import Prediction
from hierarchicalforecast.evaluation import HierarchicalEvaluation

from utils.metricas import rmse, mase
from utils.utils import format_ds
#%%
'''
Programa para gerar o dataframe com as folhas para ser usado no calculo
dos clusters . Dataset tourism.
'''
if __name__=='__main__':

    print (os.getcwd())
    RAW_DIR = '../Data_MHTS/' #'../../Data_MHTS/' mac puc '../Data_MHTS/' mac novo
    
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
    PROCESSED_DIR = 'data/processed/'
    ######### 1. Load and Process Data
    australia = Preprocessing(RAW_DIR, RAW_FILE) 
    Y_df = australia.load_preprocess_tourism()

    #todas as combinacoes de hierarquias
    spec = [
        ['Country'],
        ['Country', 'State'], 
        ['Country', 'Purpose'], 
        ['Country', 'State', 'Region'], 
        ['Country', 'State', 'Purpose'], 
        ['Country', 'State', 'Region', 'Purpose']
    ]

    spec_bottom = [        
        ['Country', 'State', 'Region', 'Purpose']
    ]

    Y_df_bottom, S_df, tags = australia.aggregate_df(Y_df, spec_bottom)

    #print ("{}\n, {}\n", Y_df_bottom.head(2), tags)

    #changing the format od column ds to be yy-mm-dd
    Y_df_bottom['ds'] = Y_df_bottom['ds'].apply(format_ds)

    #criating df pivot in the format: columns= name of the serie and days. to be used in cluster
    Y_df_bottom_pivot=Y_df_bottom.pivot_table(index='unique_id',columns='ds',values='y')
    Y_df_bottom_pivot.reset_index(inplace=True)
    print ("{}\n".format(Y_df_bottom_pivot.head(2)))

    # saving to pickle file
    print (os.getcwd())
    Y_df_bottom_pivot.to_pickle(PROCESSED_DIR+'Tourism_bottom_pivot.pkl')

    #### Save to pickle file the aggregate, needs to have Y_df2 to be used for dominio 
    #    and Y_df with unique column to be used with clusters

    spec = [
        ['Country'],
        ['Country', 'State'], 
        ['Country', 'Purpose'], 
        ['Country', 'State', 'Region'], 
        ['Country', 'State', 'Purpose'], 
        ['Country', 'State', 'Region', 'Purpose']
    ]

    Y_df2, S_df, tags = australia.aggregate_df(Y_df, spec) #precisa disso para incluir no Y_df com 
                                                           #a coluna de ids_unico

    australia.save_aggregate(Y_df2, S_df, tags, Y_df, PROCESSED_DIR, 'Tourism_bottom_pivot' )
    '''
    ##### divide in train and test
    Y_df2=Y_df_bottom
    steps = 8
    Y_test_df, Y_train_df=australia.split_test_train(Y_df2, steps)

    print (Y_test_df.shape, Y_train_df.shape, Y_df2.shape)

    
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
    #'''
