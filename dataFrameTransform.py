import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
from scipy.stats import yeojohnson
from psycopg2 import errors
from data_Transform import DataTransform
from sklearn.preprocessing import PowerTransformer 


class Data_FrameTransform:

    def __init__(self):
       pass

    def Nullremoval(self, df: pd.DataFrame, pct: int):

         '''
        This method is used to drop columns where the percentage of missing values is above an amount specified by the user.

        Parameters:
            DataFrame (pd.DataFrame): The dataframe to which this method will be applied.
            pct (int): the percentage set by the user representing the required percentage of missing values in order to drop the column

        Returns:
            DataFrame (pd.DataFrame): the updated DataFrame.
        '''
         
         #initiates a blank list of columns to remove
         toRemove = []

         #counts number of null values in each column
         null_values = df.isna().sum() 

         #calculates percentage of missing values in each column
         pcnt_null = null_values/df.shape[0] * 100
         
         #loops through columns to build the list of columns to remove
         for i in pcnt_null.index:
            if pcnt_null[i] > pct:
               toRemove.append(i)

         #removes the columns that meet the threshold
         for i in toRemove:
            df.drop(i, axis=1, inplace=True)
         print(df)

         return df
      
    def impute_null_values(self, df:pd.DataFrame):         
        '''This method imputes null values in the DataFrame.
              
        Returns:
        --------
        dataframe
            A Pandas DataFrame
        '''
        for col in df:

            if df[col].dtype == 'category':
                df[col] = df[col].fillna(df[col].mode()[0])

            elif df[col].dtype == 'float64' or df[col].dtype == 'Int64' or df[col].dtype == 'int64' or df[col].dtype==  'object':
                df[col] = df[col].fillna(df[col].median())

    def apply_skew_transform(self,df:pd.DataFrame):

        pass

    def log_transform_skewed_columns(self,df):
      
      for col in df:
        if df[col].dtype == 'float64' or df[col].dtype == 'Int64' or df[col].dtype == 'int64':
          log_col = df[col].map(lambda i: np.log(i) if i > 0 else 0)
          t=sns.histplot(log_col,label="Skewness: %.2f"%(log_col.skew()) )
          t.legend()

    def yjt_transform_skewed_columns(self, df:pd.DataFrame):
      
        cols_to_transform = df.select_dtypes(['float64','Int64','int64'])
        
        # Model Creation
        p_scaler = PowerTransformer(method='yeo-johnson')
        # yeojohnTr = PowerTransformer(standardize=True)   # not using method attribute as yeo-johnson is the default

        # fitting and transforming the model
        df_yjt = pd.DataFrame(p_scaler.fit_transform(cols_to_transform))
  
        transformed_df = df_yjt.to_csv('df_yjt.csv')
  
        return transformed_df

if __name__ == "__main__":
    

    import dataFrameInfo as dx
    import data_Transform as datatransform

    df = pd.read_csv('loan_payments.csv')

    df.rename(columns={"term": "term(mths)"},inplace=True)
    df['term(mths)'] = df['term(mths)'].str.replace("months", " ")
    
    cat_data = ['id', 'member_id','grade','sub_grade','home_ownership','verification_status','loan_status', 'purpose','application_type','employment_length']
    int_data = []
    float_data = ['term(mths)', 'mths_since_last_delinq','mths_since_last_record', 'collections_12_mths_ex_med','mths_since_last_major_derog']
    bool_data = ['payment_plan']
    date_data = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

    data = Data_FrameTransform()
    dt = datatransform.DataTransform()

    for col in cat_data:
        dt.cat_type(df,col)

    for col in int_data:
        dt.num_type(df,col)

    for col in float_data:
        dt.float_type(df,col)

    for col in bool_data:
        dt.bool_type(df,col)

    for col in date_data:
        dt.convert_dates(df,col)

    new_df = data.impute_null_values(df)

    #new_df = data.log_transform_skewed_columns(df)
    new_df = data.yjt_transform_skewed_columns(df)
