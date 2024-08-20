import pandas as pd

class Data_FrameTransform:


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
