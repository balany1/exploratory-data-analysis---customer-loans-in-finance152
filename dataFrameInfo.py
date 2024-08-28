import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from scipy.stats import zscore
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect
from pandas.api.types import is_float_dtype
from pandas.api.types import is_int64_dtype

class Data_FrameInfo:
   
   def describe_dataframe(self, df:pd.DataFrame):

      df_stats = self.set_data_frame(df)
      print(df_stats)


   def df_shape(self, self,df:pd.DataFrame):
      '''
        This method computes statistics for numerical columns in the dataframe

        Args:
        --------
           df (pd.DataFrame): The dataframe to which this method will be applied.
           

        Returns:
        --------
            df.shape (pd.DataFrame): the shape of the DataFrame.
        '''

      df = self.set_data_frame(df)
      print(df.shape)


   def store_stats(self,df:pd.DataFrame):
      
      '''
        This method computes statistics for numerical columns in the dataframe

        Args:
        --------
           df (pd.DataFrame): The dataframe to which this method will be applied.
           

        Returns:
        --------
            DataFrame (pd.DataFrame): the updated DataFrame.
        '''

      stats = df.describe(include=np.number).applymap(lambda x: f"{x:0.2f}")
      print(stats)

   def get_mean(self,df:pd.DataFrame, column: str):

      '''
      Gets mean of specified column

      Args:
      --------
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
      --------
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
      --------
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
      --------
            df (pd.DataFrame): DataFrame Object being worked on
            column (str): column for which count is required
      '''

      print(f"unique {column} values: " + str(df[column].value_counts().count()))


   def null_values(self,df):

      '''
        This method computes the percentage of null values in each column

        Args:
        --------
           df (pd.DataFrame): The dataframe to which this method will be applied.
           

        '''
      
      df = self.set_data_frame(df)
      null_values = df.isna().sum() 
      pcnt_null = null_values/df.shape[0] * 100
      print(pcnt_null)


   def get_skew_info(self, df: pd.DataFrame):
      '''
      Gets the skew value for every compatible column in the dataframe

      Args:
         df(pd.DataFrame): Dataframe being worked on
      '''

      for col in df:
         if df[col].dtype == 'float64' or df[col].dtype == 'Int64' or df[col].dtype == 'int64':
            print(f"Skew of {col} is {df[col].skew()}")

   def find_regression_line(self):
      pass
   
   def z_scores(self,df:pd.DataFrame):
      '''
        This method calculates the z-score for each numerical column in the dataframe

        Args:
        --------
           df (pd.DataFrame): The dataframe to which this method will be applied.
           

        Returns:
        --------
            df_zscore (pd.DataFrame): the updated DataFrame.
        '''
      # calculate z score for each column
      df_zscore = df.apply(zscore)
                     
      # view DataFrame
      df_zscore.head()

      return df_zscore


      

   def find_high_skew_cols(self, df:pd.DataFrame, skew_limit: int = 5):
      
      '''
      Gets the skew value for every compatible column in the dataframe

      Args:
       --------
         df(pd.DataFrame): Dataframe being worked on

      Returns:
        --------
        high_skew_cols
            A list of high skew columns
      '''

      high_skew_cols = []
      
      for col in df: 
         if df[col].dtype == 'float64' or df[col].dtype == 'Int64' or df[col].dtype == 'int64':
            if df[col].skew() > skew_limit:
               high_skew_cols.append(col)
      
      return high_skew_cols