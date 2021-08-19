# -*- coding: utf-8 -*-
"""ApplicationNote_7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iN1aDY9_OvdEXUDK8vIcAYONMRvcyFWx

# ApplicationNote_7

Submitted by: Galli, Känel, Kruta, Stalder

Task: Application of SVM in breast cancer diagnosis

Step 1: Import needed packages
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score

"""Step 2: Create upload widget for data import"""

import io
from google.colab import files
df = files.upload()
df = pd.read_csv('breast-cancer-wisconsin.csv',delimiter=";")

"""Step 3: Get an overview of data"""

df.describe

"""Step 4: Column Descriptions - Data Understanding

1. Sample code number: id number
2. Clump Thickness: 1 - 10
3. Uniformity of Cell Size: 1 - 10
4. Uniformity of Cell Shape: 1 - 10
5. Marginal Adhesion: 1 - 10
6. Single Epithelial Cell Size: 1 - 10
7. Bare Nuclei: 1 - 10
8. Bland Chromatin: 1 - 10
9. Normal Nucleoli: 1 - 10
10. Mitoses: 1 - 10
11. Class: (2 for benign, 4 for malignant)
"""

df.columns =['Nr.', 'Thick.', 'C.size', 'C.shape', 'Adh.', 'Ec.size', 'Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Class']
df.head

"""Step 5: Remove "Class" from X; Create Y of "Class""""

df = pd.get_dummies(df)
y = df["Class"]
X = df.drop(["Class"], axis = 1)
Y.head
y.head

"""Step 6: Split into test and training set"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.20)

"""Step 7: Apply SVM algorithm to training set"""

clf = svm.SVC(C=1.0, kernel = "linear")
clf.fit(X_train, y_train)

"""Step 8: Prediction"""

y_pred = clf.predict(X_test)

"""Step 9: Print Accuracy of Test and Prediction"""

print("Accuracy:" , metrics.accuracy_score(y_test, y_pred))

"""Step 10: Confusion Matrix"""

confusion_matrix(y_test, y_pred)
plot_confusion_matrix(clf, X_test, y_test)

"""Step 11: Balanced accuracy score"""

balanced_accuracy_score(y_test, y_pred)