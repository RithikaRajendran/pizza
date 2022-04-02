
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('pizza.csv')
x=df.drop(['buy'],axis=1)
y=df['buy']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, random_state = 42 )

models = [
          ('LogReg', LogisticRegression()), 
          ('KNN', KNeighborsClassifier())
         ]

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['Good', 'bad']
for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(Y_test, y_pred, target_names=target_names))

pickle.dump(clf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
model1 = pickle.load(open('model1.pkl','rb'))
#print(model.predict(sc.transform(np.array([[20,40]]))))


