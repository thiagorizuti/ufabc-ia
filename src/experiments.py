import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from recommenders import CollaborativeFiltering, ClusteringFiltering, distance_matrix

plt.style.use('seaborn')
random_state=42

train= pd.read_csv('../data/ua.base',sep='\t',header=None,names=['user_id','item_id','rating','timestamp'])
test= pd.read_csv('../data/ua.test',sep='\t',header=None,names=['user_id','item_id','rating','timestamp'])
user =  pd.read_csv('../data/u.user',sep='|',header=None,names=['user_id','age','gender','occupation','zipcode'])

train = train.merge(pd.get_dummies(user[['user_id','age','gender']],columns=['gender'],drop_first=True),how='left',on='user_id')
test = test.merge(pd.get_dummies(user[['user_id','age','gender']],columns=['gender'],drop_first=True),how='left',on='user_id')

matrix = train.pivot_table('rating','user_id','item_id').values
distances = distance_matrix(matrix)

X = train[['user_id','item_id','age','gender_M']].values
y = train['rating'].values
X_test = test[['user_id','item_id','age','gender_M']].values
y_test = test['rating'].values

mae1 = []
n_neighbors = [n for n in range(1,101)]
for n in tqdm(n_neighbors):
    collab = CollaborativeFiltering(n_neighbors=n)
    collab.fit(X,y,precomputed_distances=distances)
    y_pred = collab.predict(X_test)
    mae1.append(mean_absolute_error(y_test,y_pred))

best_n = n_neighbors[np.argmin(mae1)] 
best_mae1 = mae1[best_n]
print(best_n,best_mae1)

plt.figure(figsize=(15,7))
plt.title('Número de vizinhos x MAE - melhor MAE:  %.2f (%i vizinhos)'% (best_mae1,best_n),fontsize=20)
plt.plot(n_neighbors,mae1)
plt.yticks(fontsize=20)
plt.xticks(np.arange(0,100,5),fontsize=20)
plt.savefig('../output/collaborative.png')

plt.figure(figsize=(15,7))
plt.title('Número de vizinhos x MAE - melhor MAE:  %.2f (%i vizinhos)'% (best_mae1,best_n),fontsize=20)
plt.plot(n_neighbors[max(best_n-5,0):min(best_n+5,100)],mae1[max(best_n-5,0):min(best_n+5,100)])
plt.yticks(fontsize=20)
plt.xticks(np.arange(max(best_n-5,0),min(best_n+5,100)),fontsize=20)
plt.savefig('../output/collaborative_zoom.png')

mae2 = []
k_clusters = [n for n in range(1,101)]
for k in tqdm(k_clusters):
    collab = ClusteringFiltering(k_clusters=k,random_state=42)
    collab.fit(X,y,precomputed_distances=distances)
    y_pred = collab.predict(X_test)
    mae2.append(mean_absolute_error(y_test,y_pred))

best_k = k_clusters[np.argmin(mae2)] 
best_mae2 = mae2[best_k]
print(best_k,best_mae2)

plt.figure(figsize=(15,7))
plt.title('Número de grupos x MAE - melhor MAE:  %.2f (%i grupos)'% (best_mae2,best_k),fontsize=20)
plt.plot(n_neighbors,mae2)
plt.yticks(fontsize=20)
plt.xticks(np.arange(0,100,5),fontsize=20)
plt.savefig('../output/clustering.png')

plt.figure(figsize=(15,7))
plt.title('Número de grupos x MAE - melhor MAE:  %.2f (%i grupos)'% (best_mae2,best_k),fontsize=20)
plt.plot(n_neighbors[max(best_k-5,0):min(best_k+5,100)],mae2[max(best_k-5,0):min(best_k+5,100)])
plt.yticks(fontsize=20)
plt.xticks(np.arange(max(best_k-5,0),min(best_k+5,100)),fontsize=20)
plt.savefig('../output/clustering_zoom.png')