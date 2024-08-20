import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect

class Data_FrameInfo:
   
   def describe_dataframe(self, file_name):

      df_stats = self.set_data_frame(file_name)
      print(df_stats)


   def df_shape(self, file_name):
      #prints the shape of the dataframe
      df = self.set_data_frame(file_name)
      print(df.shape)


   def store_stats(self,df):
      #computes statistics for numerical columns

      stats = df.describe(include=np.number).applymap(lambda x: f"{x:0.2f}")
      print(stats)


   def count_distinct(self, column):
      #finds the number of unique entries for columns with categorical data


      # app_type_counts = df['application_type'].value_counts()
      # loan_status_counts = df['loan_status'].value_counts()
      # ver_status_counts = df['verification_status'].value_counts()
      # home_own_counts = df['home_ownership'].value_counts()
      # purpose_counts = df['home_ownership'].value_counts()
      # grade_counts = df['grade'].value_counts()
      # subgrade_counts = df['subgrade'].value_counts()

      # print("unique application types: " + str(app_type_counts.count()))
      # print("unique loan statuses: " + str(loan_status_counts.count()))
      # print("unique verification statuses: " + str(ver_status_counts.count()))
      # print("unique home ownership statuses: " + str(ver_status_counts.count()))
      # print("unique purposes : " + str(ver_status_counts.count()))
      # print("unique grades : " + str(ver_status_counts.count()))
      # print("unique subgrades : " + str(ver_status_counts.count()))

      print("unique f{column} values" + str(df[column].value_counts().count()))

   def null_values(self,file_name):
      df = self.set_data_frame(file_name)
      null_values = df.isna().sum() 
      pcnt_null = null_values/df.shape[0] * 100
      print(pcnt_null)