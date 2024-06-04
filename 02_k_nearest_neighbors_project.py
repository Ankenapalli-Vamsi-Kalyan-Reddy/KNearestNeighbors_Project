# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""## Get the Data
** Read the 'KNN_Project_Data csv file into a dataframe **
"""

df =pd.read_csv('KNN_Project_Data')

"""**Check the head of the dataframe.**"""

df.head()



"""# EDA

Since this data is artificial, we'll just do a large pairplot with seaborn.

**Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**
"""

sns.pairplot(df, hue='TARGET CLASS')



"""# Standardize the Variables

Time to standardize the variables.

** Import StandardScaler from Scikit learn.**
"""

from sklearn.preprocessing import StandardScaler

"""** Create a StandardScaler() object called scaler.**"""

scaler = StandardScaler()

"""** Fit scaler to the features.**"""

scaler.fit(df.drop('TARGET CLASS', axis=1))

"""**Use the .transform() method to transform the features to a scaled version.**"""

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

"""**Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**"""

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head()






