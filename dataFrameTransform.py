import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import data_Transform as datatransform
from scipy.stats import yeojohnson
from psycopg2 import errors
from data_Transform import DataTransform
from sklearn.preprocessing import PowerTransformer
from dataFrameInfo import zscore 


class Data_FrameTransform:

    def __init__(self, df: pd.DataFrame):
       
       self.df = df

    def Nullremoval(self, pct: int):

        '''
        This method is used to drop columns where the percentage of missing values is above an amount specified by the user.

        Args:
        --------
            pct (int): the percentage set by the user representing the required percentage of missing values in order to drop the column

        Returns:
        --------
            df (pd.DataFrame): the updated DataFrame.
        '''
         
         #initiates a blank list of columns to remove
        toRemove = []

         #counts number of null values in each column
        null_values = self.df.isna().sum() 

         #calculates percentage of missing values in each column
        pcnt_null = null_values/self.df.shape[0] * 100
         
         #loops through columns to build the list of columns to remove
        for i in pcnt_null.index:
            if pcnt_null[i] > pct:
               toRemove.append(i)

         #removes the columns that meet the threshold
        for i in toRemove:
            self.df.drop(i, axis=1, inplace=True)
    
        return self.df
      
    def impute_null_values(self):         
        '''
        
        This method imputes null values in the DataFrame.
              
        Returns:
        --------
        dataframe
            A Pandas DataFrame
        '''
        for col in self.df:

            if self.df[col].dtype == 'category':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            elif self.df[col].dtype == 'float64' or self.df[col].dtype == 'Int64' or self.df[col].dtype == 'int64' or self.df[col].dtype==  'object':
                self.df[col] = self.df[col].fillna(self.df[col].median())


    def log_transform_skewed_columns(self):
      
      '''
        Apply log transformation to a specified column in a DataFrame.

        Returns:
        --------
        - DataFrame: Transformed DataFrame
        '''
      for col in self.df:
        if self.df[col].dtype == 'float64' or self.df[col].dtype == 'Int64' or self.df[col].dtype == 'int64':
          log_col = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
          t=sns.histplot(log_col,label="Skewness: %.2f"%(log_col.skew()) )
          t.legend()

    def transform_column(self, column_name, method='yeo-johnson', inverse_transform=False, skew_limit: int = 2):
        
        '''
        Apply Yeo-Johnson or Box-Cox transformation to a specified column in a DataFrame.
        
        Args:
        --------
        - column_name (str): Name of the column to be transformed
        - method (str): Transformation method ('yeo-johnson' or 'box-cox')
        - inverse_transform (bool): Whether to apply the inverse transformation
        - skew_limit(int): The amount of skew above which the transformation is applied

        Returns:
        --------
        - new_df: Transformed DataFrame
        '''
        
        #check skew is less than the user defined limit

        new_df = self.df

        if new_df[column_name].skew() > abs(skew_limit):
            # Print the skewness before transformation
            print(f"Skewness of {column_name} before transformation: {new_df[column_name].skew()}")
        
            # Initialize PowerTransformer
            power_transformer = PowerTransformer(method=method)
        
             # Fit and transform the specified column
        
            new_df[[column_name]] = power_transformer.fit_transform(new_df[[column_name]])
        
        # Print the skewness after transformation
        print(f"Skewness of {column_name} after transformation: {new_df[column_name].skew()}")
        
        # Display the transformed DataFrame
        display(new_df)
        
        if inverse_transform:
            # Reverse transform the transformed DataFrame
            new_df[[column_name]] = power_transformer.inverse_transform(new_df[[column_name]])
            print(f"Skewness of {column_name} after inverse transformation: {new_df[column_name].skew()}")
            # Display the reversed transformed DataFrame
            display(new_df)

        return new_df
    
    def treat_outliers_IQR(self):
        
        ''' 
        Change outlier values to upper or lower limit values i.e. those outside of 1.5*IQR of the lower and upper quartiles
            
        Returns:
        --------
        new_df
            A Pandas DataFrame
        ''' 
        # select only the numeric columns in the DataFrame
        new_df = self.df.select_dtypes('float64')
       
        for col in new_df:
 
            q1 = new_df[col].quantile(0.25)
            q3 = new_df[col].quantile(0.75) 
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr  

            new_df.loc[new_df[col]<=lower_limit, col] = lower_limit
            new_df.loc[new_df[col]>=upper_limit, col] = upper_limit

        return new_df
    
    def treat_outliers_Zscore(self, limit:int = 3.5):
        
        '''
        This method is used to drop rows where the z-score of a datapoint is above a certain threshold

        Args:
        --------
            limit (int): the abs(z-score) set by the user above which rows are dropped. Default is 3.5

        Returns:
        --------
            DataFrame (pd.DataFrame): the updated DataFrame.
        '''
        #check shape of df before outliers removed
        #print(df.shape)

        #calculates z_scores for the dataframe

        columns = self.df.select_dtypes(include=['float64', 'int64', 'Int64'])

        z_scoresdf = zscore(columns)

        #selects all rows above the z-score limit
        rows_to_drop = z_scoresdf[z_scoresdf.apply(lambda row: row.ge(abs(limit)).any(), axis=1)]

        #drops the according rows
        df1 = self.df.drop(rows_to_drop.index) # () looks for row number, [] looks for a key

        #check shape of df after outliers removed
        print(df1.shape)

        return df1


    def remove_overcorrelation(self, limit: int = 0.9):
        '''
        Drop columns if they are over correlated
        
        Args:
        --------
        - limit(int): the threshold for dropping columns
        
        Returns:
        --------
        - new_df(DataFrame): Transformed DataFrame
        '''
        
        #select cols_from_data
        cols_to_check = self.df.select_dtypes('float64').corr()

        #initiate empty list of empty columns to drop
        cols_to_drop = []

        list_a = cols_to_check.columns

        for i in range(0,len(list_a)-1):
            for j in range(i,len(list_a)-1):
                if cols_to_check.iloc[i,j] > limit and cols_to_check.iloc[i,j] < 1:
                    if list_a[j] not in cols_to_drop:
                        cols_to_drop.append(list_a[j])

        print(cols_to_drop)
        new_df = self.df.drop(columns=cols_to_drop)

        return new_df

if __name__ == "__main__":
    
    

    df = pd.read_csv('loan_payments.csv')

    df.rename(columns={"term": "term(mths)"},inplace=True)
    df['term(mths)'] = df['term(mths)'].str.replace("months", " ")
    
    cat_data = ['id', 'member_id','grade','sub_grade','home_ownership','verification_status','loan_status', 'purpose','application_type','employment_length']
    int_data = []
    float_data = ['term(mths)', 'mths_since_last_delinq','mths_since_last_record', 'collections_12_mths_ex_med','mths_since_last_major_derog']
    bool_data = ['payment_plan']
    date_data = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

    data = Data_FrameTransform(df)
    dt = datatransform.DataTransform(df)

    for col in cat_data:
        dt.cat_type(col)

    for col in int_data:
        dt.num_type(col)

    for col in float_data:
        dt.float_type(col)

    for col in bool_data:
        dt.bool_type(col)

    for col in date_data:
        dt.convert_dates(col)

    new_df = data.impute_null_values()

    #new_df = data.log_transform_skewed_columns()


    # for col in df:
    #     if df[col].dtype == np.int64 or df[col].dtype == np.float64:
    #         new_df = data.transform_column(col, 'yeo-johnson')
    

    data2 = Data_FrameTransform(df)
    new_df = data2.treat_outliers_IQR()
    print(new_df.head())

    
    new_df = data2.treat_outliers_Zscore()
    print(new_df.head())

    new_df = data2.remove_overcorrelation()
    print(new_df.shape)



