import yaml
import sqlalchemy
import pandas as pd
import numpy as np
import missingno as msno
import dataFrameInfo as dx
from dataFrameInfo import Data_FrameInfo
from data_Transform import DataTransform
from psycopg2 import errors
from scipy import stats
from sqlalchemy import create_engine
from sqlalchemy import inspect
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import plotly.express as px
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

    def histogram(self, column_name: str):
        
        '''
        This method plots a histogram for data within a column in the dataframe.

        Args:
         -----------
            column_name (str): The name of the column for which a histogram will be plotted.
        
        Returns:
            plotly.graph_objects.Figure: A histogram plot of the data within 'column_name'.
        '''

        fig = px.histogram(self.df, column_name)
        return fig.show()
    
    def pie_chart(self, labels: list, sizes: list, title: str=None):

        '''
        This method is used to generate a bar chart plot of categorical data.

        Args:
         -----------
            labels (list): The names of the categories in a list.
            sizes (list): The respective dependant variables in a list.
            title (str): DEFAULT = None, the title of the plot.

        Returns:
         -----------
            matplotlib.pyplot.figure: a pie chart plot of the data.
        '''

        #create pie chart
        plt.pie(sizes, labels=labels, colors=['#66b3ff', '#ffff99', '#00FF00'], autopct='%1.1f%%', startangle=180) # Generate pie chart.
        if title != None: # If a title is provided.
            plt.title(title)
        #set legend for pie chart
        plt.legend()
        #show pie chart
        plt.show()

    def skewness_histogram(self, column_name: str):
        
        '''
        This method plots a histogram for data within a column in the dataframe with the skewness identified.

        Args:
         -----------
            column_name (str): The name of the column for which a histogram will be plotted.
        
        Returns:
         -----------
            matplotlib.axes._subplots.AxesSubplot: A histogram plot of the data within 'column_name' with skewness identified.
        '''

        histogram = sns.histplot(df[column_name],label="Skewness: %.2f"%(df[column_name].skew()) )
        histogram.legend()
        return histogram

    def missing_matrix(self):

        '''
        This method plots a matrix displaying missing or null data points within the DataFrame.

        Returns:
         -----------
            matplotlib.axes._subplots.AxesSubplot: A matrix plot showing all the missing or null data points in each column in white.
        '''

        return msno.matrix(self.df)

    def qqplot(self, column_name: str):

        '''
        This method is used to return a Quantile-Quantile (Q-Q) plot of a column.

        Args:
         -----------
            column_name (str): The name of the column which will be plotted.

        Returns:
         -----------
            matplotlib.pyplot.figure: a Q-Q plot of the column.
        '''

        qq_plot = qqplot(self.df[column_name] , scale=1 ,line='q') 
        return plt.show()

    def facet_grid_histogram(self, column_names: list):

        '''
        This method is used to return a Facet Grid containing Histograms with the distribution drawn for a list of columns.

        Args:
         -----------
            column_names (list): A list of names of columns which will be plotted.

        Returns:
         -----------
            facet_grid (sns.FacetGrid): A facetgrid containing the histogram plots of each of the variables.
        '''

        melted_df = pd.melt(self.df, value_vars=column_names) # Melt the dataframe to reshape it.
        facet_grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False) # Create the facet grid
        facet_grid = facet_grid.map(sns.histplot, "value", kde=True) # Map histogram onto each plot on grid.
        return facet_grid

    def facet_grid_box_plot(self, column_names: list):

        '''
        This method is used to return a Facet Grid containing box-plots for a list of columns.

        Args:
         -----------
            column_names (list): A list of names of columns which will be plotted.

        Returns:
         -----------
            facet_grid (sns.FacetGrid): A facetgrid containing the box-plots of each of the variables.
        '''

        melted_df = pd.melt(self.df, value_vars=column_names) # Melt the dataframe to reshape it.
        facet_grid = sns.FacetGrid(melted_df, col="variable",  col_wrap=3, sharex=False, sharey=False) # Create the facet grid
        facet_grid = facet_grid.map(sns.boxplot, "value", flierprops=dict(marker='x', markeredgecolor='red')) # Map box-plot onto each plot on grid.
        return facet_grid 
   

    def seenulls(self):
        '''
        Visualizes null values in a DataFrame using missingno package.
        '''
        msno.bar(self.df)

    def heatmapnulls(self ):
        '''
        Visualizes null values in a DataFrame using missingno package.
        '''
        msno.heatmap(df)

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

    def visualize_high_skew(self, high_skew_cols:list =[]):
      '''
      Visualizes skew in identified columns

      Args:
      --------
         high_skew_cols(List): list of highly skewed columns produced by the function

      '''

      for col in high_skew_cols:
         print(col)
         self.df[col].hist()
         qq_plot = qqplot(self.df[col] , scale=1 ,line='q', fit=True)
         plt.show()
 
    def compare_skewness_transformations(self, column_name: str):
        
        '''
        This method is used to return subplots showing histograms in axes[0] and Q-Q subplots in axes[1] to compare the effect of log, box-cox and yoe-johnson transformations on skewness.

        Args:
        --------
            column_name (str): The name of the column within the dataframe to which this method will be applied.

        Returns:
        --------
            matplotlib.pyplot.subplots.figure: A plot containing subplots with histograms in axes[0] and Q-Q subplots in axes[1].
        '''

        transformed_df = self.df.copy() # Create a copy of the dataframe to perform transformations.

        # Apply transformations and create new column with transformed data
        transformed_df['log_transformed'] = self.df[column_name].map(lambda x: np.log(x) if x > 0 else 0) # Log transformation applied to value in column, if value is 0 then no transformation is done and added to new column in df copy.
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            transformed_df['box_cox'] = pd.Series(stats.boxcox(df[column_name])[0]).values # Perform box-cox transformation and add values as new column in dataframe copy.
        transformed_df['yeo-johnson'] = pd.Series(stats.yeojohnson(df[column_name])[0]).values # Perform yeo-johnson transformation and add values as new column in dataframe copy.

        # Create a figure and subplots:
        if (self.df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8)) # Create a 2x4 grid.
        else: 
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8)) # Create a 2x3 grid.

        # Set titles of subplots:
        axes[0, 0].set_title('Original Histogram')
        axes[1, 0].set_title('Original Q-Q Plot')
        axes[0, 1].set_title('Log Transformed Histogram')
        axes[1, 1].set_title('Log Transformed Q-Q Plot')

        if (self.df[column_name] <= 0).values.any() == False:        
            axes[0, 2].set_title('Box-Cox Transformed Histogram')
            axes[1, 2].set_title('Box-Cox Transformed Q-Q Plot')
            axes[0, 3].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 3].set_title('Yeo-Johnson Transformed Q-Q Plot')
        else:
            axes[0, 2].set_title('Yeo-Johnson Transformed Histogram')
            axes[1, 2].set_title('Yeo-Johnson Transformed Q-Q Plot')
         
      # Add Histograms to subplots:
        sns.histplot(self.df[column_name], kde=True, ax=axes[0, 0]) # Original Histogram
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
        stats.probplot(self.df[column_name], plot=axes[1, 0]) # Original Q-Q plot
        stats.probplot(transformed_df['log_transformed'], plot=axes[1, 1]) # Log transformed
        if (df[column_name] <= 0).values.any() == False: # If column contains only positive values.
            stats.probplot(transformed_df['box_cox'], plot=axes[1, 2]) # Box Cox Q-Q plot
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 3]) # Yeo Johnson Q-Q plot
        else: # If column contains non-positive values.
            stats.probplot(transformed_df['yeo-johnson'], plot=axes[1, 2]) # Yeo Johnson Q-Q plot

        plt.suptitle(column_name, fontsize='xx-large') # Add large title for entire plot.
        plt.tight_layout() # Adjust the padding between and around subplots.
        return plt.show()
   

    def visualise_outliers(self):
        '''This method visualises the data to determine if the columns contain outliers. It uses Seaborn's Boxplot to achieve this.       

        Returns:
        --------
        plot
            Seaborn's Boxplot.
        ''' 
        #select only the numeric columns in the DataFrame
        df = self.df.select_dtypes(include=['float64'])
        plt.figure(figsize=(18,14))

        for i in list(enumerate(self.df.columns)):
            fig_cols = 4
            fig_rows = int(len(self.df.columns)/fig_cols) + 1
            plt.subplot(fig_rows, fig_cols, i[0]+1)
            sns.boxplot(data=self.df[i[1]]) 

        # Show the plot
        plt.tight_layout()
        return plt.show()
   

    def show_correlation_heatmap(self):
        '''This method visualises the collinearity of data in the dataset. It uses Seaborn's heatmap to achieve this.       



        Returns:
        --------
        plot
            Seaborn's Correlation Heatmap
        ''' 
        
        #select only the numeric columns in the DataFrame
        df = self.df.select_dtypes(include=['float64'])

        #print correlation of data
        print(df.corr())

        #generate heatmap for correlation
        plt = sns.heatmap(df.corr(), cmap="YlGnBu")

        # Show the plot
        #return plt.show()
    
    def bar_chart(self, independent_categories: list, dependent_variables: list, title: str=None, y_label: str=None, x_label: str=None):
        
        '''
        This method is used to generate a bar chart plot of categorical data.

        Args:
        --------
            independent_categories (list): The names of the categories in a list.
            dependent_variables (list): The respective dependant variables in a list.
            title (str): DEFAULT = None, the title of the plot.
            y_label (str): DEFAULT = None, the label for the y-axis.
            x_label (str): DEFAULT = None, the label for the x-axis.

        Returns:
            --------
            matplotlib.pyplot.figure: a bar plot of the data.
        '''
        plt.figure(figsize=(16, 8))
        sns.barplot(x=independent_categories, y=dependent_variables) # Generating the bar plot and setting the independant and dependant variables.
        if y_label != None: # If a 'y_label' is provided.
            plt.ylabel(y_label)
        if x_label != None: # If a 'x_label' is provided.
            plt.xlabel(x_label)
        if title != None: # If a 'title' is provided.
            plt.title(title)
        return plt.show()
        
if __name__ == "__main__":
       
      
      df = pd.read_csv('loan_payments.csv')
      plotter = Plotter(df)

      df.rename(columns={"term": "term(mths)"},inplace=True)
      df['term(mths)'] = df['term(mths)'].str.replace("months", " ")

      to_object_columns = ['id', 'member_id', 'policy_code']
      to_float_columns = ['loan_amount'] 
      to_category_columns = ['term(mths)', 'grade', 'sub_grade', 'employment_length', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']
      to_integer_columns = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'collections_12_mths_ex_med']
      to_date_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

      data = DataTransform(df)

      new_df = data.obj_type(to_object_columns)
      new_df = data.float_type(to_float_columns)
      new_df = data.cat_type(to_category_columns)
      new_df = data.float_type(to_integer_columns)
     #new_df = data.convert_dates(to_date_columns)

      #plotter.compare_skewness_transformations('annual_inc')
      #plotter.histogram('loan_amount')
      #plotter.pie_chart(df.groupby(by=['grade']).groups, df.groupby(by=['grade'])['grade'].count(), title= "Pie Chart showing distributions of bank-assigned loan grades")
      #histogram = plotter.skewness_histogram('loan_amount')
      #plotter.missing_matrix()
      #plotter.qqplot('loan_amount')
      #plotter.compare_skewness_transformations('loan_amount')
      #facet_box = plotter.facet_grid_box_plot(['loan_amount','annual_inc'])
      #facet_hist = plotter.facet_grid_histogram(['loan_amount','annual_inc'])
      #plotter.visualise_skewness()
      #plotter.heatmapnulls()
      #plotter.seenulls()
      #plotter.show_correlation_heatmap()
      #plotter.bar_chart(['loan_status'],['grade'], "Loan Status vs Grade", "Loan Status", "Grade")
      #plotter.visualise_outliers()
      

