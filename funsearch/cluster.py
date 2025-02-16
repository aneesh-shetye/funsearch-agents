def create_clusters(self):

  def dbscan(similarity_matrix, eps):

    distance_matrix = (1 - similarity_matrix).clamp(min=0)

bscan = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
            labels = dbscan.fit_predict(distance_matrix)

            return labels

    embeddings = [agent['embedding'] for agent in self.db]
    similariy_score = self.embedding_model.model.similarity(embeddings, embeddings)
    cluster_labels = dbscan(similariy_score, 0.05)
        # for each cluster find the highest scoring element only
        # keep that in the cluster
        # each cluster will have a saturated or not tag
     for cluster_label in set(cluster_labels):
        highest_score = 0
        highest_idx = 0
        for idx, label in enumerate(cluster_labels):
          if label==cluster_label and self.db[idx]['score']>highest_score:

            highest_score = self.db[idx]['score']
            highest_idx = idx
            self.cluster.append({
                "code": self.db[highest_idx]['code'], 
                "score": self.db[highest_idx]['score'], 
                "embedding": self.db[highest_idx]['embedding'],
                "saturated": False})
            #print(f"Added agent to cluster {len(self.cluster)} with score {highest_score}.")

