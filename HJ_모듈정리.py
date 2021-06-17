#군집 -Kmeans,실루엣 계수
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

#--------모델 적용
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter=200, random_state=121)
cluster_labels = kmeans.fit_predict(X)
centers = kemans.cluster_centers_
unique_labels = np.unique(cluster_labels)

for label in unique_labels:
  label_cluster = clusterDF[clusterDF['kmeans_label']==label]
  center_x_y
