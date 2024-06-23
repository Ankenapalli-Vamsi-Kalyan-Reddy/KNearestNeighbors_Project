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

"""# Train Test Split

**Use train_test_split to split your data into a training set and a testing set.**
"""

from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

"""# Using KNN

**Import KNeighborsClassifier from scikit learn.**
"""

from sklearn.neighbors import KNeighborsClassifier

"""**Create a KNN model instance with n_neighbors=1**"""

knn = KNeighborsClassifier(n_neighbors=1)

"""**Fit this KNN model to the training data.**"""

knn.fit(X_train, y_train)



"""# Predictions and Evaluations
Let's evaluate our KNN model!

**Use the predict method to predict values using your KNN model and X_test.**
"""

predictions = knn.predict(X_test)
predictions

"""** Create a confusion matrix and classification report.**"""

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))



print(classification_report(y_test, predictions))



"""# Choosing a K Value
Let's go ahead and use the elbow method to pick a good K Value!

** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**
"""

delta = []
for i in range(1, 40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  p = knn.predict(X_test)
  delta.append(np.mean(p != y_test))

plt.figure(figsize=(11, 4))
plt.plot(range(1, 40), delta, color='red', lw=1, linestyle='--', marker='o', markersize=4, markerfacecolor='green')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')



"""**Now create the following plot using the information from your for loop.**"""



"""## Retrain with new K Value

**Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**
"""

knn = KNeighborsClassifier(n_neighbors=37)
knn.fit(X_train, y_train)
p = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))







