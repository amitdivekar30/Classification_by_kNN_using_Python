# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Zoo.csv')
X = dataset.iloc[:, 1: 17]
y = dataset.iloc[:, [17]]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Z# checking suitable neighbours
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(2,50):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    train_acc = np.mean(classifier.predict(X_train)==y_train.type)
    test_acc = np.mean(classifier.predict(X_test)==y_test.type)
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(2,50),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(2,50),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])

#from plot k=5 is suitable
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# train accuracy 
train_acc = np.mean(classifier.predict(X_train)==y_train.type)

# test accuracy
test_acc = np.mean(classifier.predict(X_test)==y_test.type)

#confusion matrix
cm_train = pd.crosstab(classifier.predict(X_train),y_train.type)
cm_train
cm_test = pd.crosstab(classifier.predict(X_test),y_test.type)
cm_test