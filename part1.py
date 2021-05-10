########### 1 Classification
print('Part 1 - Classification\n')

# 1
print('Question 1')
import pandas as pd

data = pd.read_csv('/Users/rakanzabian/Documents/PYTHON/adult.csv')
data = data.drop(columns=['fnlwgt','class'])
instances = data.shape[0]
nulls = data.isnull().sum().sum()
total = (data.shape[0])*(data.shape[1])
intances_w_nulls = data.isnull().any(axis=1).sum()
print('number of instances:',instances)
print('number of missing values:',nulls)
print('fraction of missing values over all attribute values:',nulls/total)
print('number of instances with missing values:',intances_w_nulls)
print('fraction of instances with missing values over all instances:',intances_w_nulls/instances)

# 2
print('\nQuestion 2')
from sklearn.preprocessing import LabelEncoder

data_without_nulls = data.dropna().astype(str).apply(LabelEncoder().fit_transform)
    
data = data.astype(str).apply(LabelEncoder().fit_transform)
for col in data:
    print(data[col].unique())


# 3
print('\nQuestion 3')
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

classes = pd.read_csv('/Users/rakanzabian/Documents/PYTHON/adult.csv')
classes = classes.dropna().astype(str).apply(LabelEncoder().fit_transform)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
      data_without_nulls, classes['class'], test_size=0.10)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)

print('Error rate: ',1 - (clf.score(x_test,y_test)))

# 4
print('\nQuestion 4')

D = pd.read_csv('/Users/rakanzabian/Documents/PYTHON/adult.csv')
Dp = D[D.isnull().any(axis=1)]
Dp = Dp.append(D.dropna().sample(n=len(Dp)))
Dp1 = Dp.fillna('missing')
Dp2 = Dp.fillna(Dp.mode().iloc[0])

Dp1 = Dp1.astype(str).apply(LabelEncoder().fit_transform)
clfDp1 = DecisionTreeClassifier(random_state=0)
clfDp1.fit(Dp1.drop(columns=['fnlwgt','class']),Dp1['class'])

print('Error rate Dp1: ',1 - (clfDp1.score(x_test,y_test)))  

Dp2 = Dp2.astype(str).apply(LabelEncoder().fit_transform)
clfDp2 = DecisionTreeClassifier(random_state=0)
clfDp2.fit(Dp2.drop(columns=['fnlwgt','class']),Dp2['class'])

print('Error rate Dp2: ',1 - (clfDp2.score(x_test,y_test)))
