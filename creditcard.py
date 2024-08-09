import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
credit_df=pd.read_csv('C:\\Users\\Akshaya Ganesh\\Downloads\\creditcard.csv')
print(credit_df.head())
X=credit_df.drop('Class',axis=1)
y=credit_df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
X_train_re,y_train_re=smote.fit_resample(X_train,y_train)
#standarise_the_features
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X_train_re_scaled=s.fit_transform(X_train_re)
X_test_scaled=s.transform(X_test)
#random_forest_classifer
from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier(random_state=42)
r.fit(X_train_re_scaled,y_train_re)
#logistic_Regression
from sklearn.linear_model import LogisticRegression
l=LogisticRegression(random_state=42)
l.fit(X_train_re_scaled,y_train_re)
y_pred_rf=r.predict(X_test_scaled)
y_pred_lr=l.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
print("Random Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

print("Logistic Regression Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-Score:", f1_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
