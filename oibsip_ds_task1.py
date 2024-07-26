import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris=pd.read_csv('C:\\Users\\Jacob Mario Leonard\\Downloads\\IRIS.csv')
df=pd.DataFrame(iris)
print(df.head())

x = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')