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
    
    def extract_integer_from_string(self, df: pd.DataFrame, column_name: str):

        '''
        This method is used to extract integers that are contained within strings in columns.

        Parameters:
            df (pd.DataFrame): The dataframe to which this method will be applied.
            column_name (str): The name of the column to which this method will be applied.

        Returns:
            df(pd.DataFrame): The updated DataFrame.
        '''

        df[column_name] = df[column_name].str.extract('(\d+)').astype('Int32') # The first method extracts any digits from the string in the desired column
        # the second method casts the digits into the 'Int32' data type, this is because this type of integer is a nullable type of integer.
        return df
   
    def convert_dates(self, df, date_column):
      df[date_column] = pd.to_datetime(df[date_column], format="%b-%Y")
      return df
   
    def cat_type(self, df, column):
      df[column]= df[column].astype('category')

    def num_type(self, df, column):
      df[column] = df[column].astype(int)

    def float_type(self, df, column):
      df[column] = df[column].astype(float)

    def obj_type(self, df, column):
      df[column] = df[column].astype(object)

    def bool_type(self, df, column):
      df[column] = df[column].astype(bool)
   
   
    def fill_blanks(self, df, column):
      df[column]=  df[column].replace('N/A',np.NaN)
      df[column] = df[column].replace(' ',np.NaN)


if __name__ == "__main__":
    

    import dataFrameInfo as dx
    df = pd.read_csv('loan_payments.csv')

    cat_data = ['id', 'member_id','grade','sub_grade','home_ownership','verification_status','loan_status', 'purpose','application_type','employment_length']
    int_data = []
    float_data = ['term(mths)', 'mths_since_last_delinq','mths_since_last_record', 'collections_12_mths_ex_med','mths_since_last_major_derog']
    bool_data = ['payment_plan']
    date_data = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

    data = DataTransform()

    for col in cat_data:
        data.cat_type(df,col)

    for col in int_data:
        data.num_type(df,col)

    for col in float_data:
        data.float_type(df,col)

    for col in bool_data:
        data.bool_type(df,col)

    for col in date_data:
        data.convert_dates(df,col)

