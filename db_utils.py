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





if __name__ == "__main__":

   #data = RDSDatabaseConnector()
   dataframe = DataTransform()
   df = dataframe.fix_data_types('loan_payments.csv')
   #dataframe.describe_dataframe('loan_payments.csv')
   #dataframe.count_distinct('loan_payments.csv')
   #dataframe.df_shape('loan_payments.csv')
   #dataframe.store_stats('loan_payments.csv')
   #dataframe.null_values('loan_payments.csv')
   #dataframe.Nullremoval('loan_payments.csv')
   
   
   

       
    