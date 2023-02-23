import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from difflib import SequenceMatcher
from nltk import everygrams
import math
from sklearn.manifold import TSNE


import joblib


class PredictModel:

    def __init__(self,mlModel,datasetVersion,featureNumber,applyOUSamplingLabel,df_input):

        self.MODEL_FOLDER = '/home/niyaz/Documents/Code/twitter_api_backend/trained_ml'

        self.expected_columns=[
            'Name_entropy', 'Name_freq', 'Name_similarity', 'Screen_name_entropy',
            'Screen_name_freq', '_location', '_url', 'default_profile',
            'default_profile_image', 'description_length', 'diff_days', 'entities',
            'favourites_count', 'favourites_growth_rate', 'followers_count',
            'followers_friends_ratio', 'followers_growth_rate', 'friends_count',
            'friends_growth_rate', 'geo_enabled', 'has_extended_profile',
            'listed_count', 'listed_growth_rate', 'name_length',
            'num_digits_in_name', 'num_digits_in_screen_name',
            'profile_background_image_url_https', 'profile_background_tile',
            'profile_banner_url', 'profile_use_background_image',
            'screen_name_length', 'statuses_count', 'tweet_freq', 'verified',
        ]

        self.mlModel=mlModel
        self.datasetVersion=datasetVersion
        self.featureNumber=featureNumber
        self.applyOUSamplingLabel=applyOUSamplingLabel
        self.df_input = df_input

        self.predict()

    def predict(self):
        self.model=joblib.load(f"{self.MODEL_FOLDER}/{self.mlModel}/model_{self.datasetVersion}_{self.featureNumber}_{self.applyOUSamplingLabel}.joblib")
        self.df=pd.DataFrame(self.model.predict(self.df_input.values).tolist(),columns=["class"])
        #self.df['class']=self.df['class'].map({1:"bot",0:"human"})
        self.df['class']=self.df['class'].map({1:"red",0:"green"})


    def output(self):
        return self.df["class"].value_counts().to_json()

    def class_output(self):
        predict_class=self.df['class'].map({"red":"bot","green":"human"})
        return predict_class
  
    def TSNE(self):

        for i in range(len(self.model.steps)-1):
            print(type(self.model.steps[i][1]).__name__)
            if type(self.model.steps[i][1]).__name__=="QuantileTransformer":
                X=self.model.steps[i][1].transform(self.df_input.values)
                X=pd.DataFrame(X,columns=self.expected_columns)
            elif type(self.model.steps[i][1]).__name__=="SelectFromModel":
                X=X[X.columns[self.model.steps[i][1].get_support()]]

        TSNE_RESULT = TSNE(n_components=2,init="pca",learning_rate='auto').fit_transform(X)
        return pd.DataFrame({
            'X1': TSNE_RESULT[:,0], 
            "X2": TSNE_RESULT[:,1], 
            'Y':self.df['class']
        }).to_json(orient='values')