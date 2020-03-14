import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# reading the data
loans = pd.read_csv('loan_data.csv')

print(loans.info())
print(loans.head())
print(loans.describe())

# exploratory data analysis
# creating histogram of two FICO distributions:
# one for each credit.policy outcome

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5, color='blue',
                                              bins=30, label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5, color='red',
                                              bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# creating similar figure but by not.fully.paid column
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# creaing countplot using seaborn showing counts of loans by purpose with hue by not.full.paid
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
plt.show()

# creating a plot showing the trend between FICO score and interest rate
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.show()

# creating lmplot to see if the trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
plt.show()

# setting up the data

#using dummy variables tranforming 'purpose' column so sklearn will be able to understand
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.info())

# train test split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# training a decision tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# predictions and evaluation of decision tree
predictions = dtree.predict(X_test)
print('Decision Tree:')
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# training the random forest model
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

# predictions and evaluation
preds = rfc.predict(X_test)
print('Random Forest:')
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))