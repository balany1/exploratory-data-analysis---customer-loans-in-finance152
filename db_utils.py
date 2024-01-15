import yaml
import sqlalchemy
import pandas as pd
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect




class RDSDatabaseConnector:
    
    def __init__(self) -> None:

      with open('credentials.yml', 'r') as file:
         config = yaml.load(file, Loader=yaml.FullLoader)

       #make connection to specified database
      DATABASE_TYPE = config['DATABASE_TYPE']
      DBAPI = config['DBAPI']
      RDS_HOST = config['RDS_HOST']
      RDS_PASSWORD = config['RDS_PASSWORD']
      RDS_USER = config['RDS_USER']
      RDS_DATABASE = config['RDS_DATABASE']
      RDS_PORT = config['RDS_PORT']

      self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DATABASE}")
      query = 'select * from loan_payments'
      df = pd.read_sql(query,self.engine)
      df.to_csv('loan_payments.csv', index = False)

if __name__ == "__main__":

   data = RDSDatabaseConnector()

       
    