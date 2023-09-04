import pandas as pd
import numpy as np

import pickle

from statsforecast.models import ETS
from statsforecast.core import StatsForecast

#3. Reconcile forecasts
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.core import HierarchicalReconciliation
#to aggregate data in hierarchies
#from hierarchicalforecast.utils import aggregate

class Prediction:
    def __init__(self):
        self.reconcilers = [
            BottomUp(),
            MinTrace(method='mint_shrink'),
            MinTrace(method='ols')
        ]
        self.season_length = 4
        self.model = 'ZZA'

    def predict_ets(self, Y_train_df, s_length, md, fq, h):
        if not s_length:
            season_length = self.season_length
        if not md:
            md = self.model

        print ("predict_ets: ", s_length, md, fq, h)
        fcst = StatsForecast(df=Y_train_df, 
                            models=[ETS(season_length=s_length, model=md)], 
                            freq=fq, n_jobs=-1)
        Y_hat_df = fcst.forecast(h, fitted=True)
        Y_fitted_df = fcst.forecast_fitted_values()

        return Y_hat_df, Y_fitted_df
    
    def rec_BU_MinTrace(self, Y_hat_df, Y_fitted_df, S_df, tags):
        hrec = HierarchicalReconciliation(reconcilers=self.reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)

        return Y_rec_df
    
    def save_pred(self):
        '''
        '''