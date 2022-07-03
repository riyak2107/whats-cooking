#Food and ingredients prediction 

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

rf=RandomForestClassifier(random_state=1)
dt=DecisionTreeClassifier(random_state=0)
nb=MultinomialNB()

d_c = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french' ,'greek', 'indian' ,'irish', 'italian' ,'jamaican' ,'japanese' ,'korean' ,'mexican' ,'moroccan', 'russian' ,  'spanish' ,'southern_us'
,'thai', 'vietnamese']

df=pd.read_json("C:/Users/Riya/Documents/train.json")
#print(df)
df['cuisine'].unique()
x=df['ingredients']
y=df['cuisine'].apply(d_c.index)

#print(x)
#print(y)

df['all_ingredients']=df['ingredients'].map(";".join)

cv=CountVectorizer()
x=cv.fit_transform(df['all_ingredients'].values)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)

dt.fit(x_train,y_train)
y_dtp=dt.predict(x_test)

nb.fit(x_train,y_train)
y_nbp=nb.predict(x_test)

print('Random forest : ' ,accuracy_score(y_test,y_rfp)*100)
print('Decision Tree : ' ,accuracy_score(y_test,y_dtp)*100)
print('Naive Bayes : ' ,accuracy_score(y_test,y_nbp)*100)


#Over sampling method using

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)

dt.fit(x_train,y_train)
y_dtp=dt.predict(x_test)

nb.fit(x_train,y_train)
y_nbp=nb.predict(x_test)
print('After feature eng Random forest : ' ,accuracy_score(y_test,y_rfp)*100)
print('After feature eng Decision Tree : ' ,accuracy_score(y_test,y_dtp)*100)
print('After feature eng Naive Bayes : ' ,accuracy_score(y_test,y_nbp)*100)


''' 
OUTPUT :
Random forest :  75.48711502199875
Decision Tree :  63.70835952231301
Naive Bayes :  72.08045254556883

After feature eng Random forest :  97.96185251339628
After feature eng Decision Tree :  96.02896146976269
After feature eng Naive Bayes :  71.89334013779025

Process finished with exit code 0
'''
