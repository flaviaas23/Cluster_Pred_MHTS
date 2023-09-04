from preprocessing import Preprocessing 

#from hierarchicalforecast.utils import aggregate

if __name__=='__main__':

    RAW_DIR = '../../Data_MHTS/'
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

    #1. Load and Process Data
    australia = Preprocessing(RAW_DIR,RAW_FILE )
    
    Y_df =australia.load_preprocess_tourism()

    spec = [
        ['Country'],
        ['Country', 'State'], 
        ['Country', 'Purpose'], 
        ['Country', 'State', 'Region'], 
        ['Country', 'State', 'Purpose'], 
        ['Country', 'State', 'Region', 'Purpose']
    ]
    print (Y_df.head())
    Y_df2, S_df, tags = australia.aggregate_df(spec, Y_df)

    Y_test_df, Y_train_df = australia.split_test_train(Y_df2, h=8)

    quit()
    #2. Computing base forecasts
    h=8
    fcst = StatsForecast(df=Y_train_df, 
                     models=[ETS(season_length=4, model='ZZA')], 
                     freq='QS', n_jobs=-1)
    Y_hat_df = fcst.forecast(h=h, fitted=True)
    Y_fitted_df = fcst.forecast_fitted_values()

#Entendendo o dataset de tourism da australia para gerar
#  as funcoes e hierarquias
#1. Load and Process Data
#%%
import pandas as pd
import numpy as np
import os

#%%
from preprocessing import Preprocessing 


#%%
print (os.getcwd())
RAW_DIR = '../../../../Data_MHTS/'
#RAW_FILE = 'tourism.csv'
RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

#%%
######### 1. Load and Process Data
australia = Preprocessing() #RAW_DIR, RAW_FILE)
australia.raw_dir=RAW_DIR
australia.raw_file=RAW_FILE
Y_df=australia.load_preprocess_tourism()
#%%
spec = [
    ['Country'],
    ['Country', 'State'], 
    ['Country', 'Purpose'], 
    ['Country', 'State', 'Region'], 
    ['Country', 'State', 'Purpose'], 
    ['Country', 'State', 'Region', 'Purpose']
]

#%%
Y_df2, S_df, tags = australia.aggregate_df(Y_df, spec)



#%%
steps = 8
Y_test_df, Y_train_df=australia.split_test_train(Y_df2, steps)
#2. Computing base forecasts
# %%
from statsforecast.models import ETS
from statsforecast.core import StatsForecast
#%%
fcst = StatsForecast(df=Y_train_df, 
                     models=[ETS(season_length=4, model='ZZA')], 
                     freq='QS', n_jobs=-1)
Y_hat_df = fcst.forecast(h=8, fitted=True)
Y_fitted_df = fcst.forecast_fitted_values()

#%%
#este codigo nao esta ok
from predict import Prediction
steps = 8
season_length = 4
model = 'ZZA'
freq = 'QS'

pred = Prediction()
Y_hat_df, Y_fitted_df= pred.predict_ets(Y_train_df, s_length = season_length, 
                                        md = model, fq = freq, h = steps
                                        )

# %%
### 3. Reconcile forecasts

from predict import Prediction

pred = Prediction()
Y_rec_df= pred.rec_BU_MinTrace(Y_hat_df, Y_fitted_df, S_df, tags)
# %%
#4. Evaluation
from hierarchicalforecast.evaluation import HierarchicalEvaluation



#%%
def rmse(y, y_hat):
    return np.mean(np.sqrt(np.mean((y-y_hat)**2, axis=1)))

def mase(y, y_hat, y_insample, seasonality=4):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    return np.mean(errors / scale)

#%%
eval_tags = {}
eval_tags['Total'] = tags['Country']
eval_tags['Purpose'] = tags['Country/Purpose']
eval_tags['State'] = tags['Country/State']
eval_tags['Regions'] = tags['Country/State/Region']
eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
eval_tags['All'] = np.concatenate(list(tags.values()))
#%%
eval_tags = {}
eval_tags['Total'] = [x for x in tags['Country']]
eval_tags['Purpose'] = [x for x in tags['Country/Purpose']]
eval_tags['State'] = [x for x in tags['Country/State']]
eval_tags['Regions'] = [x for x in tags['Country/State/Region']]
eval_tags['Bottom'] = [x for x in tags['Country/State/Region/Purpose']]
eval_tags['All'] = np.concatenate(list(tags.values()))
#%%
evaluator = HierarchicalEvaluation(evaluators=[rmse, mase])
#%%
evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
        tags=eval_tags, Y_df=Y_train_df
)
#%%
evaluation2 = evaluation.drop('Overall')
evaluation2.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']
evaluation2 = evaluation2.applymap('{:.2f}'.format)
# %%
