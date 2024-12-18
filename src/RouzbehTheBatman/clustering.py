def steps():
    steps = """
    1- Impute the missing values ()
    2- Normalize the data (normalize)
    3- Choose the algorithm (algorithms)
    """
    print(steps)


def normalize():
    normalize = """
    from sklearn.preprocessing import StandardScaler

    scaler_standard = StandardScaler() # A.K.A. Z-score
    scaler_standard = MinMaxScaler() # min-max scaler
    scaler_standard = MaxAbsScaler() # Scale each feature to the [-1, 1] range without breaking the sparsity
    

    df_normalized = scaler.fit_transform(df)
    """
    
    print(normalize)

def elbow():
    process = """
    #Libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = pd.read_csv("path")

    cluster_range = range(2, 60)
    inertia_values = []
    silhouette_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(df)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, clusters))

    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid()
    plt.show()


    # Silhouette scores plot
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker = 'o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Cluster Counts')
    plt.grid()
    plt.show()
    """
    print(process)