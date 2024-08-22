import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from dataFrameInfo import Data_FrameInfo
from psycopg2 import errors
from sqlalchemy import create_engine
from sqlalchemy import inspect
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import seaborn as sns

class Plotter:
      
   '''
    This class contains the methods which are used to visualise insights from the data.

    Parameters:
    -----------
    data_frame: DataFrame
        A Pandas DataFrame from which information will be generated.

    Methods:
    --------
    visualise_nulls_impute()
        Visualises the data to check if all the null values have been imputed.
    
    visualise_outliers()
        Visualises the data to determine if the columns contain outliers.
    '''    
   def __init__(self, data_frame) -> None:
      self.df = data_frame
   
   def seenulls(self, df):

      msno.bar(df)

   def heatmapnulls(self, df):
      
      msno.heatmap(df)

   def impute_nulls(self,df):
      pass

   def visualise_skewness(self):
      '''
      
      This method plots the data to visualise the skew. It uses Seaborn's Histogram with KDE line plot to achieve this.       
            
      Returns:
      --------
      plot
         Seaborn's Histogram with KDE line plot.
      '''  
      #select only the numeric columns in the DataFrame
      df = self.df.select_dtypes(include=['float64'])
      plt.figure(figsize=(18,14))

      for i in list(enumerate(df.columns)):
         fig_cols = 4
         fig_rows = int(len(df.columns)/fig_cols) + 1
         plt.subplot(fig_rows, fig_cols, i[0]+1)
         sns.histplot(data = df[i[1]], kde=True)

      # Show the plot
      plt.tight_layout()
      return plt.show()

   def visualize_high_skew(self, df, high_skew_cols:list =[]):
      '''
      Visualizes skew in identified columns

      Args:
         df(pd.DataFrame): dataframe being worked on
         high_skew_cols(List): list of highly skewed columns produced by the function
      '''

      for col in high_skew_cols:
         print(col)
         df[col].hist()
         qq_plot = qqplot(df[col] , scale=1 ,line='q', fit=True)
         plt.show()

   