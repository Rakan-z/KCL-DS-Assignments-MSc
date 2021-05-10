########### 2 Clustering
print('\nPart 2 - Clustering\n')


# 1
print('Question 1')
import pandas as pd

data2 = pd.read_csv('/Users/rakanzabian/Documents/PYTHON/wholesale_customers.csv')
data2 = data2.drop(columns=['Channel','Region'])

mean = []
mini = []
maxi = []
i=-1
for col in data2:
    i+=1
    mean.append(data2[col].mean())
    mini.append(data2[col].min())
    maxi.append(data2[col].max())
    print('Mean of {}:'.format(col), mean[i])
    print('Range of {}:'.format(col), maxi[i]-mini[i])
    
# 2
import sklearn.cluster as cluster
import matplotlib.pyplot as plt

km = cluster.KMeans(n_clusters=3)
km.fit(data2)

LABEL_COLOR_MAP = {0 : 'r',
                    1 : 'k',
                    2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in km.labels_]
plt.figure()
plt.scatter(data2['Fresh'], data2['Milk'], c=label_color)
plt.title('Fresh vs. Milk')
plt.xlabel('Fresh')
plt.ylabel('Milk')

plt.figure()
plt.scatter(data2['Fresh'], data2['Grocery'], c=label_color)
plt.title('Fresh vs. Grocery')
plt.xlabel('Fresh')
plt.ylabel('Grocery')

plt.figure()
plt.scatter(data2['Fresh'], data2['Frozen'], c=label_color)
plt.title('Fresh vs. Frozen')
plt.xlabel('Fresh')
plt.ylabel('Frozen')

plt.figure()
plt.scatter(data2['Fresh'], data2['Detergents_Paper'], c=label_color)
plt.title('Fresh vs. Detergents_Paper')
plt.xlabel('Fresh')
plt.ylabel('Detergents_Paper')

plt.figure()
plt.scatter(data2['Fresh'], data2['Delicassen'], c=label_color)
plt.title('Fresh vs. Delicassen')
plt.xlabel('Fresh')
plt.ylabel('Delicassen')

plt.figure()
plt.scatter(data2['Milk'], data2['Grocery'], c=label_color)
plt.title('Milk vs. Grocery')
plt.xlabel('Milk')
plt.ylabel('Grocery')

plt.figure()
plt.scatter(data2['Milk'], data2['Frozen'], c=label_color)
plt.title('Milk vs. Frozen')
plt.xlabel('Milk')
plt.ylabel('Frozen')

plt.figure()
plt.scatter(data2['Milk'], data2['Detergents_Paper'], c=label_color)
plt.title('Milk vs. Detergents_Paper')
plt.xlabel('Milk')
plt.ylabel('Detergents_Paper')

plt.figure()
plt.scatter(data2['Milk'], data2['Delicassen'], c=label_color)
plt.title('Milk vs. Delicassen')
plt.xlabel('Milk')
plt.ylabel('Delicassen')

plt.figure()
plt.scatter(data2['Grocery'], data2['Frozen'], c=label_color)
plt.title('Grocery vs. Frozen')
plt.xlabel('Grocery')
plt.ylabel('Frozen')

plt.figure()
plt.scatter(data2['Grocery'], data2['Detergents_Paper'], c=label_color)
plt.title('Grocery vs. Detergents_Paper')
plt.xlabel('Grocery')
plt.ylabel('Detergents_Paper')

plt.figure()
plt.scatter(data2['Grocery'], data2['Delicassen'], c=label_color)
plt.title('Grocery vs. Delicassen')
plt.xlabel('Grocery')
plt.ylabel('Delicassen')

plt.figure()
plt.scatter(data2['Frozen'], data2['Detergents_Paper'], c=label_color)
plt.title('Frozen vs. Detergents_Paper')
plt.xlabel('Frozen')
plt.ylabel('Detergents_Paper')

plt.figure()
plt.scatter(data2['Frozen'], data2['Delicassen'], c=label_color)
plt.title('Frozen vs. Delicassen')
plt.xlabel('Frozen')
plt.ylabel('Delicassen')

plt.figure()
plt.scatter(data2['Detergents_Paper'], data2['Delicassen'], c=label_color)
plt.title('Detergents_Paper vs. Delicassen')
plt.xlabel('Detergents_Paper')
plt.ylabel('Delicassen')


# 3
print('\nQuestion 3')
import numpy as np
 
WC = km.inertia_
between = np.zeros((3))
for i in range(3):
    between[i] = 0.0
    for l in range(i+1,3): 
        for n in range(6):
            between[i]+=np.square(km.cluster_centers_[i][n]-km.cluster_centers_[l][n])
BC = np.sum(between)
print('3-\n','WC:',WC,'\nBC:',BC,'\nBC/WC:',BC/WC)

km2 = cluster.KMeans(n_clusters=5)
km2.fit(data2)
WC2 = km2.inertia_
between2 = np.zeros((5))
for i in range(5):
    between2[i] = 0.0
    for l in range(i+1,5): 
        for n in range(6):
            between2[i]+=np.square(km2.cluster_centers_[i][n]-km2.cluster_centers_[l][n])
BC2 = np.sum(between2)
print('5-\n','WC:',WC2,'\nBC:',BC2,'\nBC/WC:',BC2/WC2)

km3 = cluster.KMeans(n_clusters=10)
km3.fit(data2)
WC3 = km3.inertia_
between3 = np.zeros((10))
for i in range(10):
    between3[i] = 0.0
    for l in range(i+1,10): 
        for n in range(6):
            between3[i]+=np.square(km3.cluster_centers_[i][n]-km3.cluster_centers_[l][n])
BC3 = np.sum(between3)
print('10-\n','WC:',WC3,'\nBC:',BC3,'\nBC/WC:',BC3/WC3)
