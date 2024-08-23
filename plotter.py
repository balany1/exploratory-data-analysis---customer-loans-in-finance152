import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
from dataFrameInfo import Data_FrameInfo
from data_Transform import DataTransform
from psycopg2 import errors
from scipy import stats
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
   
   def __init__(self, df:pd.DataFrame) -> None:
      self.df = df
   
   def seenulls(self, df):

      msno.bar(df)

   def heatmapnulls(self, df):
      
      msno.heatmap(df)

   def impute_nulls(self,df):
      pass

   def visualise_skewness(self, df):
      '''
      
      This method plots the data to visualise the skew. It uses Seaborn's Histogram with KDE line plot to achieve this.       
            
      Returns:
      --------
      plot
         Seaborn's Histogram with KDE line plot.
      '''  
      #select only the numeric columns in the DataFrame
      df = df.select_dtypes(include=['float64'])
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
 
   def compare_skewness_transformations(self, df: pd.DataFrame, column_name: str):
        
        '''
        This method is used to return subplots showing histograms in axes[0] and Q-Q subplots in axes[1] to compare the effect of log, box-cox and yoe-johnson transformations on skewness.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe to which this method will be applied.
            column_name (str): The name of the column within the dataframe to which this method will be applied.

        Returns:
            matplotlib.pyplot.subplots.figure: A plot containing subplots with histograms in axes[0] and Q-Q subplots in axes[1].
        '''

        transformed_df = df.copy() # Create a copy of the dataframe to perform transformations.

        # Apply transformations and create new column with transformed data
        transformed_df['log_transformed'] = df[column_name].map(lambda x: np.log(x) if x > 0 else 0) # Log transformation applied to value in column, if value is 0 then no transformation is done and added to new column in df copy.
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            transformed_df['box_cox'] = pd.Series(stats.boxcox(df[column_name])[0]).values # Perform box-cox transformation and add values as new column in dataframe copy.
        transformed_df['yeo-johnson'] = pd.Series(stats.yeojohnson(df[column_name])[0]).values # Perform yeo-johnson transformation and add values as new column in dataframe copy.

        # Create a figure and subplots:
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8)) # Create a 2x4 grid.
        else: 
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8)) # Create a 2x3 grid.

        # Set titles of subplots:
        axes[0, 0].set_title('Original Histogram')
        axes[1, 0].set_title('Original Q-Q Plot')
        axes[0, 1].set_title('Log Transformed Histogram')
        axes[1, 1].set_title('Log Transformed Q-Q Plot')
        if (df[column_name] <= 0).values.any() == False:        
            axes[0, 2].set_title('Box-Cox Transformed Histogram')
            axes[1, 2].set_title('Box-Cox Transformed Q-Q Plot')
            axes[0, 3].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 3].set_title('Yeo-Johnson Transformed Q-Q Plot')
        else:
            axes[0, 2].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 2].set_title('Yeo-Johnson Transformed Q-Q Plot')
         
      # Add Histograms to subplots:
        sns.histplot(df[column_name], kde=True, ax=axes[0, 0]) # Original Histogram
        axes[0, 0].text(0.5, 0.95, f'Skewness: {df[column_name].skew():.2f}', ha='center', va='top', transform=axes[0, 0].transAxes) # Add skewness
        sns.histplot(transformed_df['log_transformed'], kde=True, ax=axes[0, 1]) # Log transformed Histogram
        axes[0, 1].text(0.5, 0.95, f'Skewness: {transformed_df["log_transformed"].skew():.2f}', ha='center', va='top', transform=axes[0, 1].transAxes) # Add skewness
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            sns.histplot(transformed_df['box_cox'], kde=True, ax=axes[0, 2]) # Box Cox Histogram
            axes[0, 2].text(0.5, 0.95, f'Skewness: {transformed_df["box_cox"].skew():.2f}', ha='center', va='top', transform=axes[0, 2].transAxes) # Add skewness
            sns.histplot(transformed_df['yeo-johnson'], kde=True, ax=axes[0, 3]) # Yeo Johnson Histogram
            axes[0, 3].text(0.5, 0.95, f'Skewness: {transformed_df["yeo-johnson"].skew():.2f}', ha='center', va='top', transform=axes[0, 3].transAxes) # Add skewness
        else: # If column contains non-positive values.
            sns.histplot(transformed_df['yeo-johnson'], kde=True, ax=axes[0, 2]) # Yeo Johnson Histogram
            axes[0, 2].text(0.5, 0.95, f'Skewness: {transformed_df["yeo-johnson"].skew():.2f}', ha='center', va='top', transform=axes[0, 2].transAxes) # Add skewness

        # Add Q-Q plots to subplots:
        stats.probplot(df[column_name], plot=axes[1, 0]) # Original Q-Q plot
        stats.probplot(transformed_df['log_transformed'], plot=axes[1, 1]) # Log transformed
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            stats.probplot(transformed_df['box_cox'], plot=axes[1, 2]) # Box Cox Q-Q plot
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 3]) # Yeo Johnson Q-Q plot
        else: # If column contains non-positive values.
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 2]) # Yeo Johnson Q-Q plot

        plt.suptitle(column_name, fontsize='xx-large') # Add large title for entire plot.
        plt.tight_layout() # Adjust the padding between and around subplots.
        return plt.show()
    
if __name__ == "__main__":
       
     
      import dataFrameInfo as dx
      df = pd.read_csv('loan_payments.csv')
      plotter = Plotter(df)

      to_object_columns = ['id', 'member_id', 'policy_code']
      to_float_columns = ['loan_amount'] 
      to_category_columns = ['term', 'grade', 'sub_grade', 'employment_length', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']
      to_integer_columns = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'collections_12_mths_ex_med']
      to_date_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

      data = DataTransform()

      new_df = data.obj_type(df, to_object_columns)
      new_df = data.float_type(df, to_float_columns)
      new_df = data.cat_type(df, to_category_columns)
      new_df = data.float_type(df, to_integer_columns)
      #new_df = data.convert_dates(df, to_date_columns)

      #plotter.compare_skewness_transformations(df, 'annual_inc')
