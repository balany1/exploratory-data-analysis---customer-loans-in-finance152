import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect
from pandas.api.types import is_float_dtype
from pandas.api.types import is_int64_dtype

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

   def get_mean(self,df:pd.DataFrame, column: str):

      '''
      Gets mean of specified column

      Args:
         df(pd.DataFrame) : dataframe containing the column required
         column(str): Column to be aggreagted
      '''

      if is_float_dtype(df[column]) == True or is_int64_dtype(df[column]) == True:
         mean = df.loc[:, column].mean()

      print("%.2f"% mean)

   def get_median(self,df:pd.DataFrame, column: str):
      
      '''
      Gets median of specified column

      Args:
         df(pd.DataFrame) : dataframe containing the column required
         column(str): Column to be aggreagted
      '''

      if is_float_dtype(df[column]) == True or is_int64_dtype(df[column]) == True:
         median = df.loc[:, column].median()

      print("%.2f"% median)

   def get_sd(self,df:pd.DataFrame, column: str):
      '''
      Gets standard deviation of specified column

      Args:
         df(pd.DataFrame) : dataframe containing the column required
         column(str): Column to be aggreagted
      '''

      if is_float_dtype(df[column]) == True or is_int64_dtype(df[column]) == True:
         stdev = df.loc[:, column].std()

      print("%.2f"% stdev)


   def count_distinct(self, df: pd.DataFrame, column: str):
      '''
      Counts number of distinct values in a particular column and returns it

      Args:
            df (pd.DataFrame): DataFrame Object being worked on
            column (str): column for which count is required
      '''

      print(f"unique {column} values: " + str(df[column].value_counts().count()))


   def null_values(self,file_name):
      df = self.set_data_frame(file_name)
      null_values = df.isna().sum() 
      pcnt_null = null_values/df.shape[0] * 100
      print(pcnt_null)