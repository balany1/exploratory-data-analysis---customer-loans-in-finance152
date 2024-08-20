import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect

class Plotter:
      
      def seenulls(self, df):

         msno.bar(df)

      def heatmapnulls(self, df):
         
         msno.heatmap(df)

      def impute_nulls(self,df):
         pass