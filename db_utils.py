import yaml
import sqlalchemy
import pandas as pd
import numpy as np
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect




class RDSDatabaseConnector:
    
    def __init__(self) -> None:

      with open('credentials.yml', 'r') as file:
         config = yaml.load(file, Loader=yaml.FullLoader)

      #import credentials from yaml file
      DATABASE_TYPE = config['DATABASE_TYPE']
      DBAPI = config['DBAPI']
      RDS_HOST = config['RDS_HOST']
      RDS_PASSWORD = config['RDS_PASSWORD']
      RDS_USER = config['RDS_USER']
      RDS_DATABASE = config['RDS_DATABASE']
      RDS_PORT = config['RDS_PORT']

      #make connection to specified database
      self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DATABASE}")
      query = 'select * from loan_payments'
      df = pd.read_sql(query,self.engine)
      df.to_csv('loan_payments.csv', index = False)

class DataTransform:
   
   def __init__(self) -> None:
      
      pass
   
   def set_data_frame(self, file_name):
      
      #read_csv_file
      df = pd.read_csv(file_name)
      #print(df.dtypes)
      return df
   
   def fix_data_types(self, file_name):
      
      df = self.set_data_frame(file_name)
      pd.set_option('display.max_rows', 43)
      df['id'] = df['id'].astype('object')
      df['member_id'] = df['member_id'].astype('object')
      df['term'].str.replace(" months ", "")
      df.rename(columns={"term": "term(mths)"})
      df['issue_date'] = pd.to_datetime(df['issue_date'],dayfirst=False, format="mixed")
      df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'],dayfirst=False, format="mixed")

      df['mths_since_last_delinq'].replace('N/A',np.NaN)
      df['mths_since_last_record'].replace('N/A',np.NaN)
      df['collections_12_mths_ex_med'].replace('N/A',np.NaN) 
      df['mths_since_last_major_derog'].replace('N/A',np.NaN)

      df['mths_since_last_delinq'].replace(' ',np.NaN)
      df['mths_since_last_record'].replace(' ',np.NaN)
      df['collections_12_mths_ex_med'].replace(' ',np.NaN) 
      df['mths_since_last_major_derog'].replace(' ',np.NaN)

      df['mths_since_last_delinq'] = df['mths_since_last_delinq'].astype('float64')
      df['mths_since_last_record'] = df['mths_since_last_record'].astype('float64')
      df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].astype('float64')
      df['mths_since_last_major_derog'] = df['mths_since_last_major_derog'].astype('float64')

      df['last_payment_date'] = pd.to_datetime(df['last_payment_date'],dayfirst=False, format="mixed")
      df['next_payment_date'] = pd.to_datetime(df['next_payment_date'],dayfirst=False, format="mixed")
      df['last_credit_pull_date'] = pd.to_datetime(df['last_credit_pull_date'],dayfirst=False, format="mixed")
      
      #print(df.dtypes)
      #print(df)
      #return df

class DataFrameInfo:
   
   def describe_dataframe(self, file_name):

      df_stats = self.set_data_frame(file_name)
      print(df_stats)

   def df_shape(self, file_name):
      #prints the shape of the dataframe
      df = self.set_data_frame(file_name)
      print(df.shape)

   def store_stats(self,file_name):
      #computes statistics for numerical columns

      df = self.set_data_frame(file_name)
      # for column in df:
      #    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
      #       quant_cols.append(column)
      
      stats = df.describe(include=np.number).applymap(lambda x: f"{x:0.2f}")
      print(stats)

   def count_distinct(self,file_name):
      #finds the number of unique entries for columns with categorical data

      df = self.set_data_frame(file_name)

      app_type_counts = df['application_type'].value_counts()
      loan_status_counts = df['loan_status'].value_counts()
      ver_status_counts = df['verification_status'].value_counts()
      home_own_counts = df['home_ownership'].value_counts()
      purpose_counts = df['home_ownership'].value_counts()
      grade_counts = df['grade'].value_counts()
      subgrade_counts = df['subgrade'].value_counts()

      print("unique application types: " + str(app_type_counts.count()))
      print("unique loan statuses: " + str(loan_status_counts.count()))
      print("unique verification statuses: " + str(ver_status_counts.count()))
      print("unique home ownership statuses: " + str(ver_status_counts.count()))
      print("unique purposes : " + str(ver_status_counts.count()))
      print("unique grades : " + str(ver_status_counts.count()))
      print("unique subgrades : " + str(ver_status_counts.count()))

   def null_values(self,file_name):
      df = self.set_data_frame(file_name)
      null_values = df.isna().sum() 
      pcnt_null = null_values/df.shape[0] * 100
      print(pcnt_null)


class DataFrameTransform:
      
      def set_data_frame(self, file_name):
      
         #read_csv_file
         df = pd.read_csv(file_name)
         #print(df.dtypes)
         return df

      def null_values(self,file_name):
         df = self.set_data_frame(file_name)
         null_values = df.isna().sum() 
         pcnt_null = null_values/df.shape[0] * 100
         print(pcnt_null)


      def Nullremoval(self):
         pass
         

class Plotter:
      pass
# Any other methods you may find useful


if __name__ == "__main__":

   #data = RDSDatabaseConnector()
   dataframe = DataTransform()
   #df = dataframe.fix_data_types('loan_payments.csv')
   #dataframe.describe_dataframe('loan_payments.csv')
   #dataframe.count_distinct('loan_payments.csv')
   #dataframe.df_shape('loan_payments.csv')
   #dataframe.store_stats('loan_payments.csv')
   dataframe.null_values('loan_payments.csv')
   
   
   

       
    