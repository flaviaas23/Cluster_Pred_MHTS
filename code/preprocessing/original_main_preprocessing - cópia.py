from preprocessing import Preprocessing 



if __name__=='__main__':
    RAW_DIR = '../../Data_MHTS/'
    #RAW_FILE = 'tourism.csv'
    RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'

    australia = Preprocessing()
    australia.raw_dir = RAW_DIR
    australia.raw_file = RAW_FILE
    Y_df =australia.load_preprocess_tourism()

    print (df_Y.head())


#Entendendo o dataset de tourism da australia para gerar
#  as funcoes e hierarquias
#1. Load and Process Data
#%%
import pandas as pd
import numpy as np
import os

#%%
from preprocessing import Preprocessing 

australia = Preprocessing()
#%%
RAW_DIR = '../../../../Data_MHTS/'
#RAW_FILE = 'tourism.csv'
RAW_FILE = 'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
#%%
Y_df = pd.read_csv(RAW_DIR+RAW_FILE)
#Y_df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
Y_df = Y_df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
Y_df.insert(0, 'Country', 'Australia')
Y_df = Y_df[['Country', 'Region', 'State', 'Purpose', 'ds', 'y']]
Y_df['ds'] = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
#%%
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

# %%
from hierarchicalforecast.utils import aggregate
# %%
Y_df2, S_df, tags = aggregate(Y_df, spec)
Y_df2 = Y_df2.reset_index()
#%%
from preprocessing import Preprocessing.aggregate_df
Y_df2, S_df, tags = australia.agregate_df(spec)
# %%
Y_test_df = Y_df2.groupby('unique_id').tail(8)
Y_train_df = Y_df2.drop(Y_test_df.index)

Y_test_df = Y_test_df.set_index('unique_id')
Y_train_df = Y_train_df.set_index('unique_id')
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

# %%
#3. Reconcile forecasts
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.core import HierarchicalReconciliation
#%%
reconcilers = [
    BottomUp(),
    MinTrace(method='mint_shrink'),
    MinTrace(method='ols')
]

hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)

# %%
#4. Evaluation
from hierarchicalforecast.evaluation import HierarchicalEvaluation

#%%
#nao esta funcionando ...
from metricas import rmse, mase

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
