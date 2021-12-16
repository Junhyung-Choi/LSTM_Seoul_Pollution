import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib 


class PriceDataset():
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.
    
    def __init__(self):
        scaler = RobustScaler()
        
        train_df = pd.read_csv("./dataset/price_data_tr.csv")
        val_df = pd.read_csv("./dataset/price_data_val.csv")
        
        train_df = train_df.drop(["date"], axis=1)
        val_df = val_df.drop(["date"], axis=1)

        train_df.iloc[:,2:] = scaler.fit_transform(train_df.iloc[:,2:])
        val_df.iloc[:,2:] = scaler.transform(val_df.iloc[:,2:])

        tr_x = train_df.iloc[:,2:].values
        tr_y = train_df.iloc[:,1].values
        
        val_x = val_df.iloc[:,2:].values
        val_y = val_df.iloc[:,1].values
        
        joblib.dump(scaler, './models/scaler.pkl') 
        
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.val_x = val_x
        self.val_y = val_y
    
    def getDataset(self):
        return [self.tr_x, self.tr_y, self.val_x, self.val_y]