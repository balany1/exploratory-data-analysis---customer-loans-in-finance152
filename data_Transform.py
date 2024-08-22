import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
from scipy.stats import yeojohnson
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect


class DataTransform:
   
    def __init__(self) -> None:
      
      pass
   
    def set_data_frame(self, file_name):
      
      #read_csv_file
      df = pd.read_csv(file_name)
      #print(df.dtypes)
      return df
    
    def extract_integer_from_string(self, DataFrame: pd.DataFrame, column_name: str):

        '''
        This method is used to extract integers that are contained within strings in columns.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe to which this method will be applied.
            column_name (str): The name of the column to which this method will be applied.

        Returns:
            DataFrame (pd.DataFrame): The updated DataFrame.
        '''

        DataFrame[column_name] = DataFrame[column_name].str.extract('(\d+)').astype('Int32') # The first method extracts any digits from the string in the desired column
        # the second method casts the digits into the 'Int32' data type, this is because this type of integer is a nullable type of integer.
        return DataFrame
   
    def convert_dates(df, date_column):
      df[date_column] = pd.to_datetime(df[date_column], format="%b-%Y")
      return df
   
    def cat_type(df, column):
      df[column]= df[column].astype('category')

    def num_type(df, column):
      df[column] = df[column].astype(int)

    def float_type(df, column):
      df[column] = df[column].astype(float)

    def obj_type(df, column):
      df[column] = df[column].astype(object)

    def bool_type(df, column):
      df[column] = df[column].astype(bool)
   
   
    def fill_blanks(df, column):
      df[column]=  df[column].replace('N/A',np.NaN)
      df[column] = df[column].replace(' ',np.NaN)

    def log_transform_skewed_columns(self,df):
      log_population = df["Population"].map(lambda i: np.log(i) if i > 0 else 0)
      t=sns.histplot(log_population,label="Skewness: %.2f"%(log_population.skew()) )
      t.legend()

    def yjt_transform_skewed_columns(self,df):
      yeojohnson_population = df["Population"]
      yeojohnson_population = yeojohnson(yeojohnson_population)
      yeojohnson_population= pd.Series(yeojohnson_population[0])
      t=sns.histplot(yeojohnson_population,label="Skewness: %.2f"%(yeojohnson_population.skew()) )
      t.legend()
    