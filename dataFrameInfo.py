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

   def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
      self.df = df
   
   def describe_dataframe(self):

      df_stats = self.df.describe()
      print(df_stats)


   def df_shape(self):
      '''
        This method computes statistics for numerical columns in the dataframe

        Returns:
        --------
            df.shape (pd.DataFrame): the shape of the DataFrame.
        '''

      print(self.df.shape)


   def store_stats(self):
      
      '''
      This method computes statistics for numerical columns in the dataframe

      Returns:
      --------
         DataFrame (pd.DataFrame): the updated DataFrame.
      '''
      '''This method extracts statistical values: median, standard deviation and mean from the columns of the DataFrame.
      
      Returns:
      --------
      data
         A dataset of median, standard deviation and mean of all the columns in the DataFrame
      '''

      # Get all dataframe columns with float and integer data types
      columns = self.df.select_dtypes(include=['float64', 'int64', 'Int64']).columns
      # Define an empty list of rows
      rows_list = []
      # Populate list of rows
      for col in columns:
         rows_list.append([col, round(self.df[col].median(),2), round(self.df[col].std(),2), round(self.df[col].mean(),2)])
      
      # Convert the list into dataframe rows
      data = pd.DataFrame(rows_list)
      # Add column headers
      data.columns = ['column', 'median', 'std', 'mean']  
      return data
      

   def get_mean(self, column: str):

      '''
      Gets mean of specified column

      Args:
      --------
         column(str): Column to be aggreagted
      '''

      if self.df[column].dtype == np.int64 or self.df[column].dtype == np.float64:
         mean = self.df.loc[:, column].mean()

      print("%.2f"% mean)

   def get_median(self, column: str):
      
      '''
      Gets median of specified column

      Args:
      --------
   
         column(str): Column to be aggreagted
      '''

      if self.df[column].dtype == np.int64 or self.df[column].dtype == np.float64:
         median = self.df.loc[:, column].median()

      print("%.2f"% median)

   def get_sd(self, column: str):
      '''
      Gets standard deviation of specified column

      Args:
      --------
         column(str): Column to be aggreagted
      '''

      if self.df[column].dtype == np.int64 or self.df[column].dtype == np.float64:
         stdev = self.df.loc[:, column].std()

      print("%.2f"% stdev)


   def count_distinct(self, column: str):
      '''
      Counts number of distinct values in a particular column and returns it

      Args:
      --------
            column (str): column for which count is required
      '''

      print(f"unique {column} values: " + str(self.df[column].value_counts().count()))


   def null_values(self):

      '''
        This method computes the percentage of null values in each column

        Args:
        --------
           df (pd.DataFrame): The dataframe to which this method will be applied.
   
        '''
      
      
      null_values = self.df.isna().sum() 
      pcnt_null = null_values/self.df.shape[0] * 100
      print(pcnt_null)


   def get_skew_info(self):
      '''
      Gets the skew value for every compatible column in the dataframe

      '''

      for col in self.df:
         if self.df[col].dtype == 'float64' or self.df[col].dtype == 'Int64' or self.df[col].dtype == 'int64':
            print(f"Skew of {col} is {self.df[col].skew()}")

   def find_regression_line(self):
      pass
   
   def z_scores(self):
      '''
        This method calculates the z-score for each numerical column in the dataframe

        Returns:
        --------
            df_zscore (pd.DataFrame): the updated DataFrame.
        '''
      # Select datatypes for which it is possible to calculate a zscore
      columns = self.df.select_dtypes(include=['float64', 'int64', 'Int64'])
      
      # calculate z score for each column
      df_zscore = columns.apply(zscore)
                     
      # view DataFrame
      df_zscore.head()

      return df_zscore


      

   def find_high_skew_cols(self, skew_limit: int = 5):
      
      '''
      Gets the skew value for every compatible column in the dataframe

      Args:
       --------
         skew_limit(int): The limit of skewness above which the transformations are to be applied

      Returns:
        --------
        high_skew_cols
            A list of high skew columns
      '''

      high_skew_cols = []
      
      for col in self.df: 
         if self.df[col].dtype == 'float64' or self.df[col].dtype == 'Int64' or self.df[col].dtype == 'int64':
            if self.df[col].skew() > skew_limit:
               high_skew_cols.append(col)
      
      return high_skew_cols
   
if __name__ == "__main__":
   
   df = pd.read_csv('loan_payments.csv')


   dataframeinfo = Data_FrameInfo(df)

   dataframeinfo.describe_dataframe()
   dataframeinfo.df_shape()
   dataframeinfo.store_stats()
   dataframeinfo.get_mean('loan_amount')
   dataframeinfo.get_median('loan_amount')
   dataframeinfo.get_sd('loan_amount')
   dataframeinfo.count_distinct('loan_amount')
   dataframeinfo.null_values()
   dataframeinfo.get_skew_info()
   dataframeinfo.z_scores()
   high_skew_cols = dataframeinfo.find_high_skew_cols()
   print(high_skew_cols)

