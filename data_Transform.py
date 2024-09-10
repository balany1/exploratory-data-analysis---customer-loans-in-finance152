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
   
    def __init__(self, df: pd.DataFrame) -> None:
      self.df = df
    
   
    def convert_dates(self, date_column: str):
      '''
        This method is used to convert designated columns to mmm-yyyy date format

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
      self.df[date_column] = pd.to_datetime(self.df[date_column], format="%b-%Y")
      return self.df
   
    def cat_type(self, column:str):
      '''
        This method is used to esignated columns to category type

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
      self.df[column]= self.df[column].astype('category')

    def num_type(self, column:str):
       '''
        This method is used to esignated columns to integer type

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
       self.df[column] = self.df[column].astype(int)


    def float_type(self, column:str):
       '''
        This method is used to esignated columns to float type

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
       self.df[column] = self.df[column].astype(float)

    def obj_type(self, column:str):
      '''
        This method is used to designated columns to object type

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
      self.df[column] = self.df[column].astype(object)

    def bool_type(self, column:str):
      '''
        This method is used to designated columns to bool type

        Args:
        --------
            df (pd.DataFrame): The dataframe to which this method will be applied.
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
      self.df[column] = self.df[column].astype(bool)
   
   
    def fill_blanks(self, column:str):

       '''
        This method is used to fill any blank values with numpy NaN enabling it to be converted to a different type

        Args:
        --------
            column_name (str): The name of the column to which this method will be applied.

        Returns:
         --------
            df(pd.DataFrame): The updated DataFrame.
        '''
       self.df[column]=  self.df[column].replace('N/A',np.NaN)
       self.df[column] = self.df[column].replace(' ',np.NaN)


if __name__ == "__main__":
    

    import dataFrameInfo as dx
    df = pd.read_csv('loan_payments.csv')

    cat_data = ['id', 'member_id','grade','sub_grade','home_ownership','verification_status','loan_status', 'purpose','application_type','employment_length']
    int_data = []
    float_data = ['term(mths)', 'mths_since_last_delinq','mths_since_last_record', 'collections_12_mths_ex_med','mths_since_last_major_derog']
    bool_data = ['payment_plan']
    date_data = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

    df.rename(columns={"term": "term(mths)"},inplace=True)
    df['term(mths)'] = df['term(mths)'].str.replace("months", " ")

    data = DataTransform(df)

    for col in cat_data:
        data.cat_type(col)

    for col in int_data:
        data.num_type(col)

    for col in float_data:
        data.float_type(col)

    for col in bool_data:
        data.bool_type(col)

    for col in date_data:
        data.convert_dates(col)

    data.fill_blanks('loan_amount')
