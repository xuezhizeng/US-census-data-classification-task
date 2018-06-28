
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

df= pd.read_csv('census_income_learn.csv')
df.columns=range(1,43)
df.rename(columns={42: 'income'}, inplace=True)


weights=df[25]
y_train=pd.get_dummies(df['income'])
y_train.drop(columns=' - 50000.',inplace=True) #this column is now redundant

df2= pd.read_csv('census_income_test.csv')
df2.columns=range(1,43)
y_test=pd.get_dummies(df2[42])
y_test.drop(columns=' - 50000.',inplace=True)


plt.xlabel('age')
df.loc[df['income']==' - 50000.', 1].plot.hist(color='chocolate',bins=40,title='Histogram of Age')
df.loc[df['income']==' 50000+.', 1].plot.hist(color='darkmagenta',bins=40)
plt.legend(['Under 50k income','Over 50k income'])

df.groupby('income').hist(alpha=0.4) #_plot of the histograms of the numerical variable


# data correlation now !!
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
df[3]=df[3].astype('object')
df[4]=df[4].astype('object')
df[37]=df[37].astype('object')
df[39]=df[39].astype('object')

df2[3]=df2[3].astype('object')
df2[4]=df2[4].astype('object')
df2[37]=df2[37].astype('object')
df2[39]=df2[39].astype('object')

corr=pd.get_dummies(df[[3,9]],drop_first=True).corr(method='pearson', min_periods=1) #correlation() matrix

#data cleaning
df[17]=df[17]-df[18]
df2[17]=df2[17]-df2[18]#net capital
x_train=df.drop(columns=[6,7,9,10,15,18,21,22,23,25,26,27,28,30,32,33,34,35,36,38,41,'income'])
x_test=df2.drop(columns=[6,7,9,10,15,18,21,22,23,25,26,27,28,30,32,33,34,35,36,38,41,42])
print(df.isnull().sum())#non are equal to zero
print(df2.isnull().sum())# none are equal to zero

x_train=pd.get_dummies(x_train,drop_first=True)#171 columns
x_test=pd.get_dummies(x_test,drop_first=True)#171 columns
x_train.info()


print(cross_val_score(LogisticRegression(n_jobs=-1),x_train, y_train, cv=10, scoring='accuracy'))#10 folds cross validation scores
print(cross_val_score(RandomForestClassifier(n_jobs=-1,n_estimators=20),x_train, y_train, cv=10, scoring='accuracy'))

model1=LogisticRegression(n_jobs=-1)
model2=RandomForestClassifier(n_estimators=100,n_jobs=-1)

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)


predictions1=model1.predict(x_test)
predictions2=model2.predict(x_test)

print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test, predictions1))
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test, predictions1))

fpr,tpr,thresholds=roc_curve(y_train,model1.predict_proba(x_train)[:,1]) #roc_auc_score curves

associatedprediction=[ 1 if (predictions1[k]==1 or predictions2[k]==1) else 0 for k in range(len(predictions2))] #easy ensembling  of the two models in order to increase the number of true positives
print(confusion_matrix(y_test,associatedprediction))
print(classification_report(y_test, associatedprediction))

for k in range(len(thresholds)):
    if fpr[k]>0.1:
        thresh=thresholds[k]
        break

probas=model1.predict_proba(x_test)[:,1]
true_pos=sum([(probas[k]>thresh)*(y_test.values[k]==1) for k in range(len(probas))])
false_pos=sum([(probas[k]>thresh)*(y_test.values[k]==0) for k in range(len(probas))])
false_neg=sum([(probas[k]<=thresh)*(y_test.values[k]==1) for k in range(len(probas))])
true_neg=sum([(probas[k]<=thresh)*(y_test.values[k]==0) for k in range(len(probas))])

accuracy=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
recall=true_pos/(true_pos+false_neg)
precision=true_pos/(true_pos+false_pos)
sensitivity=true_neg/(true_neg+false_pos)