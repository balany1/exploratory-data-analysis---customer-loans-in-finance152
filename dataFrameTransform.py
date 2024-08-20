import pandas as pd

class Data_FrameTransform:


      def Nullremoval(self, df: pd.dataframe, pct: int):

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
