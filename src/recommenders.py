import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans



def similarity(x1,x2):
    x1=np.nan_to_num(x1)
    x2=np.nan_to_num(x2)
    mask = np.logical_and(x1>0,x2>0)
    user1, user2 = x1[mask], x2[mask]
    weight = min(1,np.sum(mask)/50)
    corr, _ = np.abs(pearsonr(x1,x2))
    return corr*weight

def distance_matrix(X):
    distance_matrix = np.zeros((X.shape[0],X.shape[0]))
    for i,x1, in enumerate(X):
        for j,x2 in enumerate(X):
            distance_matrix[i,j]= similarity(x1,x2)
    return distance_matrix    


class CollaborativeFiltering:
    

    def __init__(self,n_neighbors=5):
        
        self.n_neighbors=n_neighbors
        self.rating_matrix = None
        self.user_idx = None
        self.item_map = None
        self.item_idx=None
    
                        
    def fit(self,X,y,precomputed_distances=None):
        
        rows, row_pos = np.unique(X[:, 0], return_inverse=True)
        cols, col_pos = np.unique(X[:, 1], return_inverse=True)
        self.rating_matrix = np.full((len(rows), len(cols)), np.nan)
        self.rating_matrix[row_pos, col_pos] = y
        
        self.user_idx = {idd:idx for idd, idx in zip(rows,np.arange(rows.size))}
        self.item_idx = {idd:idx for idd, idx in zip(cols,np.arange(cols.size))}
        
        if precomputed_distances is None:
            self.distance_matrix=calculate_distance_matrix(self.rating_matrix)
        else:
            self.distance_matrix=precomputed_distances
        
    def predict(self,X):
        
        y_pred = np.full(X.shape[0],np.nan)

        users = np.unique(X[:,0])
        new_users = np.setdiff1d(users,np.fromiter(self.user_idx.keys(),dtype=int))

        for user in users: 
            
            if user in new_users: 
                predictions = np.nanmean(collab.rating_matrix,axis=0)
                
            else:
                uidx = self.user_idx[user]
                neighbors = np.argsort(self.distance_matrix[uidx,:])[-1*self.n_neighbors-1:-1]
                weights = (self.distance_matrix[uidx,neighbors])

                user_ratings = self.rating_matrix[uidx,:]
                user_mean = np.nanmean(user_ratings)

                neighbors_ratings = self.rating_matrix[neighbors]
                neighbors_mean = np.nanmean(neighbors_ratings,axis=1)

                predictions = np.nansum((np.transpose(neighbors_ratings) - neighbors_mean) * weights,axis=1)
                predictions = predictions/np.sum(weights) + user_mean

            items = X[X[:,0]==user,1]
            new_items = np.setdiff1d(items,np.fromiter(self.item_idx.keys(),dtype=int))
            
            for item in items:
                
                if item in new_items:
                    y_pred[((X[:,0]==user)&(X[:,1]==item))] = user_mean
                    
                else:
                    y_pred[((X[:,0]==user)&(X[:,1]==item))] = predictions[self.item_idx[item]]
   
        return y_pred


class ClusteringFiltering:
    
    def __init__(self,k_clusters=5,random_state=0):
        
        self.k_clusters=k_clusters
        self.random_state=random_state
        self.kmeans=None
        self.features=None
        self.rating_matrix = None
        self.user_idx = None
        self.item_idx = None
        self.distance_matrix=None
    
                        
    def fit(self,X,y,precomputed_distances=None):
        
        rows, row_pos = np.unique(X[:, 0], return_inverse=True)
        cols, col_pos = np.unique(X[:, 1], return_inverse=True)
        self.rating_matrix = np.full((len(rows), len(cols)), np.nan)
        self.rating_matrix[row_pos, col_pos] = y
        
        self.user_idx = {idd:idx for idd, idx in zip(rows,np.arange(rows.size))}
        self.item_idx = {idd:idx for idd, idx in zip(cols,np.arange(cols.size))}
                
        if precomputed_distances is None:
            self.distance_matrix=calculate_distance_matrix(self.rating_matrix)
        else:
            self.distance_matrix=precomputed_distances
                
        _, uniques = np.unique(X[:,0],return_index=True)
        self.features=minmax_scale(X[uniques,2:].astype(float))
        self.kmeans = KMeans(n_clusters=self.k_clusters,random_state=self.random_state)
        self.clusters = self.kmeans.fit_predict(self.features)
        
    def predict(self,X):
        
        y_pred = np.full(X.shape[0],np.nan)

        users = np.unique(X[:,0])
        new_users = np.setdiff1d(users,np.fromiter(self.user_idx.values(),dtype=int))
        
        _, uniques = np.unique(X[:,0],return_index=True)
        features=minmax_scale(X[uniques,2:].astype(float))
    
        for user in users: 
            
            if user in new_users: 
                predictions = np.nanmean(self.rating_matrix,axis=0)
                
            else:
                uidx = self.user_idx[user]       
                
                user_ratings = self.rating_matrix[uidx,:]
                user_mean = np.nanmean(user_ratings)
                
                cluster = self.kmeans.predict(features[uidx,:].reshape(1, -1))[0]
                cluster_members = np.fromiter(self.user_idx.values(),dtype=int)[np.where(self.kmeans.labels_==cluster)]
                
                neighbors = np.argsort(self.distance_matrix[uidx,cluster_members])
                weights = (self.distance_matrix[uidx,neighbors])

                neighbors_ratings = self.rating_matrix[neighbors]
                neighbors_mean = np.nanmean(neighbors_ratings,axis=1) 

                predictions = np.nansum((np.transpose(neighbors_ratings) - neighbors_mean) * weights,axis=1)
                if np.sum(weights)!=0:
                    predictions = predictions/np.sum(weights) + user_mean
                else:
                    predictions = np.full(predictions.shape,user_mean)

            items = X[X[:,0]==user,1]
            new_items = np.setdiff1d(items,np.fromiter(self.item_idx.keys(),dtype=int))
            
            for item in items:
                if item in new_items:
                    y_pred[((X[:,0]==user)&(X[:,1]==item))] = user_mean    
                else:
                    y_pred[((X[:,0]==user)&(X[:,1]==item))] = predictions[self.item_idx[item]]
            if np.isnan(y_pred[((X[:,0]==user)&(X[:,1]==item))]):
                print(self.distance_matrix[uidx,neighbors])
                break
   
        return y_pred



