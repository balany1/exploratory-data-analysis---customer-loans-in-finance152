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
      df['issue_date'] = pd.to_datetime(df['issue_date']).dt.date
      df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line']).dt.date
      #df['mths_since_last_delinq'] = df['mths_since_last_delinq'].astype('int64')
      #df['mths_since_last_record'] = df['mths_since_last_record'].astype('int64')
      df['last_payment_date'] = pd.to_datetime(df['last_payment_date']).dt.date
      df['next_payment_date'] = pd.to_datetime(df['next_payment_date']).dt.date
      df['last_credit_pull_date'] = pd.to_datetime(df['last_credit_pull_date']).dt.date
      #df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].astype('int64')
      print(df)

      


if __name__ == "__main__":

   #data = RDSDatabaseConnector()
   dataframe = DataTransform()
   df = dataframe.fix_data_types('loan_payments.csv')
   
   

       
    