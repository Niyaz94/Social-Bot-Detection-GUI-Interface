import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from difflib import SequenceMatcher
from nltk import everygrams
import math


class PreProcessing:

  def __init__(self,df):

    self.df = df

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

    self.df.columns = map(str.lower, self.df.columns)


    if "created_at" in self.df.columns:
      self.df['diff_days']=self.find_diff_days().replace(0, 1)
      self.create_numeric_value()
      
    self.convert_string_value()
    self.convert_bool_value()
    self.handle_nan_value()
    self.clean_other_value()
    self.drop_columns()

  def likehood(self,s):
    Bi_list=["".join(item) for item in list(everygrams(s, 2, 2))]
    K=sum(list({i:Bi_list.count(i) for i in Bi_list}.values()))
    CBi= len(np.unique(Bi_list, return_counts=True)[1])
    return CBi/K if K>0 else 1

  def entropy(self,string):
      prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
      entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
      return entropy/len(string) if len(string)>0 else 1

  def similar(self,a, b):
      return SequenceMatcher(None, a, b).ratio()

  def drop_columns(self):
      list_of_drop_column=[
        "lang","time_zone","follow_request_sent","utc_offset","notifications","following","protected","contributors_enabled","is_translator","is_translation_enabled",
        "profile_sidebar_fill_color","profile_link_color","profile_sidebar_border_color","profile_background_color","profile_text_color","id","id_str","collected_at","created_at",
        "withheld_in_countries","translator_type","profile_background_image_url","profile_image_url_https""profile_image_url","url","location","name","description","screen_name"
      ]
      self.df.drop(columns=[column_name for column_name in self.df.columns if column_name in list_of_drop_column],axis=1,inplace=True)
      self.df.drop(columns=np.setdiff1d(self.df.columns, self.expected_columns).tolist(),axis=1,inplace=True)

  def find_diff_days(self):
      return ((pd.to_datetime(self.df['collected_at']) if "collected_at" in self.df.columns else datetime.now(tz=timezone.utc))-pd.to_datetime(self.df['created_at'])).dt.days

  def handle_nan_value(self):
    for column in ["profile_background_image_url_https","profile_banner_url","profile_background_image_url","url","location"]:
      if column in self.df.columns:
        if column in ["url","location"]:
          self.df[f"_{column}"]=np.where(self.df[column].isnull(),0,1)
        else:
          self.df[f"{column}"]=np.where(self.df[column].isnull(),0,1)

  def convert_bool_value(self):
    for column in ["profile_background_tile","default_profile","profile_use_background_image","default_profile_image","geo_enabled","verified","has_extended_profile"]:
      if column in self.df.columns:
          self.df[column]= self.df[column].map({True: 1, False: 0})

  def convert_string_value(self):
    if "name" in self.df.columns:
      self.df["name"].fillna("",inplace=True)
      self.df['name_length']=self.df['name'].map(lambda x: len(x))
      self.df['num_digits_in_name']=self.df['name'].map(lambda x: sum(c.isdigit() for c in x))
      self.df['Name_freq']=self.df.apply(lambda row: self.likehood(str(row['name'])) if len(row['name'])>0 else 1,axis=1)
      self.df['Name_entropy']=self.df.apply(lambda row: self.entropy(str(row['name'])),axis=1)
    if "screen_name" in self.df.columns:
      self.df['Screen_name_entropy']=self.df.apply(lambda row: self.entropy(str(row['screen_name'])),axis=1)
      self.df['Screen_name_freq']=self.df.apply(lambda row: self.likehood(str(row['screen_name'])),axis=1)
      self.df['screen_name_length']=self.df['screen_name'].map(lambda x: len(str(x)))
      self.df['num_digits_in_screen_name']=self.df['screen_name'].map(lambda x: sum(c.isdigit() for c in str(x)))
    if "name" in self.df.columns and "screen_name" in self.df.columns:
      self.df['Name_similarity']=self.df.apply(lambda row: self.similar(str(row['screen_name']),str(row['name'])),axis=1)
    if "description" in self.df.columns:
      self.df["description"].fillna("",inplace=True)
      self.df['description_length']=self.df['description'].map(lambda x: len(x))

  def create_numeric_value(self):
    diff_days=self.df['diff_days']
    self.df['tweet_freq']              =self.df['statuses_count']/diff_days
    self.df['followers_growth_rate']   =self.df['followers_count']/diff_days
    self.df['friends_growth_rate']     =self.df['friends_count']/diff_days
    self.df['favourites_growth_rate']  =self.df['favourites_count']/diff_days
    self.df['listed_growth_rate']      =self.df['listed_count']/diff_days
    self.df['followers_friends_ratio'] =self.df['followers_count']/diff_days
    
  def clean_other_value(self):
    if "entities" in self.df.columns:
      self.df["entities"]=np.where(self.df["entities"]=="{'description': {'urls': []}}",0,1)
    if "entities" in self.df.columns:
      self.df["has_extended_profile"]=np.where(self.df["has_extended_profile"].isnull(),False,self.df["has_extended_profile"])

  def output(self):
    return self.df[self.expected_columns]
  
  def check_output(self):
    return [np.setdiff1d(self.df.columns, self.expected_columns).tolist(),np.setdiff1d(self.expected_columns,self.df.columns).tolist()]
