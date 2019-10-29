"""
utility function for working with DataFrame
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pandas_ml as pdml

TEST_DF = pd.DataFrame([1,3,4,5,6])

ReportNull = pd.isnull().sum()

#return train, validate, test of a single dataframe.
#of the dateset.
def train_val_test_permutation_split(df, train_percent, validate_percent, seed=None):

    np.random.seed(seed)
    perm = np.random.permutation(df.index) #perm will randomly reorganize the rows
    m = len(df.index) #grab the number of row in the dataframe

    #find out number of row to train, validate, use the reminder for test
    train_end = int(train_percent * m)   # find out how many rows to use to train
    validate_end = int(validate_percent * m) + train_end #find out how many rows to validate

    #create the train, validate, test variables then return.
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]

    return train,validate,test

#creating a confusion matrix

#function take takes in y_actual, and y_predicted then spits out a
#crosstab, which is turned into a heatmap;
### need to pass in list of y_actual and y_predicted!
#this is for numerical data
def confusion_Matrix_HeatMap_numerical(y_actual, y_predicted):
    #create the dataframe from y_actual and y_predicted
    data = {'y_Predicted':y_predicted,
            'y_Actual':y_actual
            }
    df = pd.DataFrame(data,columns=['y_Actual','y_Predicted'])
    #crosstab the confusion_matrix
    crosstab_confusion_matrix = pd.crosstab(df['y_Actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
    #heatmap the confusion_matrix
    heatmap = sns.heatmap(crosstab_confusion_matrix, annot=True)




