import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Social_Network_Ads.csv')

df.head()


X=df.iloc[:,0:2].values
y=df.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)


classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))