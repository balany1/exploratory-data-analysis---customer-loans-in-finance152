# exploratory-data-analysis---customer-loans-in-finance152
# Exploratory Data Analysis: Customer Loans in Finance
By **balany1**
## Table of Contents:
- [Description](#description)
    - [Key Insights](#key-insights)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [File Structure](#file-structure)
    - [File Description](#understanding-the-files)
- [Project Documentation](#project-documentation)

At minimum, your README file should contain the following information:

## Description

This is a project designed by AiCore to manage and gain insights from a dataset of loan payments. The aim of the project was to further consolidate content learned during the AICore bootcamp by preparing industry standard analysis required for a Data Analyst role.


This involved securely downloading the dataset from a PostgreSQL relational database and writing it to a csv file ready to be worked on with Pandas.

Then, the data was converted to a Pandas Dataframe and cleaned by first correcting data types of the columns in the dataset and then imputing any null values/dropping columns with little data. The missingno package was used to determine which columns needed to be dropped and how to impute the missing data. The dataset was further reduced by removing any outliers (I chose to do with a method that looked for z-scores of above 3.5) and dropping any columns that both showed a significant correlation and weren't crucial to the analysis. This was done with seaborn and matlib packages.

A method was also developed to normalise the dataset using seaborn and scipy packages to perform Yeo-Johnson transformations on the dataset.

Analysis was performed on the final dataset giving such example illustrations as below.

![sample2](Analysis_Examples/Sampleanalysis2.png)
![sample3](Analysis_Examples/Sampleanalysis3.png)
![sample4](Analysis_Examples/Sampleanalysis4.png)

### Key Insights

- Around 90% of the amount loaned out has been recovered although when considering interest payments expected this drops to 70% of the money expected that has been paid back so far.
- The bank should begin to make a profit in about 6 months time provided only a small number of customers start to default on their payments.
- Those who obtained loans for the purpose of debt consolidation were the most likely to default on their repayments.
- Outright home owners were far less likely to default on their loan payments than those paying mortgages or renting their home.
- When a loan was charged off, about 50% of the original expected recovery amount was actually recovered.

## Installation instructions

1) Clone GitHub repository: git clone https://github.com/Gits0L/exploratory-data-analysis---customer-loans-in-finance152.git

2) Navigate into the project directory: cd exploratory-data-analysis---customer-loans-in-finance152

3) Install the required packages with pip install requirements.txt

The following files are required (as well some credentials not provided here) for the project to work

db_utils.py
dataFrameInfo.py
dataFrameTransform.py
data_Transform.py
plotter.py
eda.ipynb
analysis.ipynb

## Usage instructions

To run the project simply open the eda.ipynb and analysis.ipynb files with any interactive python notebook program and go through the various steps in the notebook.
File structure of the project

## License information

MIT license