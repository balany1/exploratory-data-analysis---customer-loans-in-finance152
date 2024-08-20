import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect


class RDSDatabaseConnector:
   
    
   def __init__(self) -> None:

      '''
      Extracts credentials from Yaml file and sets up variables for use in creating an engine
      '''

      with open('credentials.yml', 'r') as file:
         config = yaml.load(file, Loader=yaml.FullLoader)

      #import credentials from yaml file
      self.DATABASE_TYPE = config['DATABASE_TYPE']
      self.DBAPI = config['DBAPI']
      self.RDS_HOST = config['RDS_HOST']
      self.RDS_PASSWORD = config['RDS_PASSWORD']
      self.RDS_USER = config['RDS_USER']
      self.RDS_DATABASE = config['RDS_DATABASE']
      self.RDS_PORT = config['RDS_PORT']

   
   def create_engine(self):

      '''
      Makes connection to Database
      '''

      #make connection to specified database
      self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.RDS_USER}:{self.RDS_PASSWORD}@{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}")
   
   
   def create_df(self):


      '''
      Creates dataframe from required table

      Returns #:

         df (pd.DataFrame): The dataframe to be saved

      '''

      query = 'select * from loan_payments'
      df = pd.read_sql(query, self.engine)
      return df


   def save_to_csv(self, df: pd.DataFrame, file_name):

      '''
      Saves DataFrame to required csv

      Args:
            df(pd.DataFrame): The dataframe to be saved
            file_name (str): The file_name given to save the DataFrame
      '''
      df.to_csv(file_name, index = False)
      

if __name__ == "__main__":

   data = RDSDatabaseConnector()
   data.create_engine()
   df = data.create_df()
   data.save_to_csv(df, 'loan_payments2.csv')
  
  