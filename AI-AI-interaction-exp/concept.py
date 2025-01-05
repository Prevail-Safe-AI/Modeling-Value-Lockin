# Evaluate the concept embedding 

'''
Three aspects to be worked on:
1/ frequency: most frequent words, their frequency, comapred to default frequency, after cleaning
2/ embedding: all the words/topics, whether  they would cluster into fewer groups over time
3/ relationships: how do new concepts appear and what are their relationshops to existing concepts 
4/ Different from the convo-concept extraction, there is also an "order" in our KB. Do we want to analyze that, too? Or it's implicitly evaluated since what's on the KB is what are with the high order?

- basic statistics: one chart for basic statistics in methdology
- the other chart 

'''
# Import 
import time
import numpy as np
import pandas as pd 
import json
import os
import sys
from typing import List, Dict, Tuple, Mapping, Union
from typeguard import check_type

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # Clustering algos 
from sklearn.metrics import silhouette_score # to decide best k in k-means 
import umap
import matplotlib.pyplot as plt 
import seaborn as sns
import voyageai

# Adjust the import search path to include its parent and parent-parent folders
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path


EMB_DIM = 256



# Embedding: str --> vector representations for later nummerical processing 
def embedding(vo: voyageai.Client, knowledges: List[List[str]]) -> List[np.array]: 
    '''
    :param knowledges: non-trivial concepts extracted from knowledge base
    :type: List

    :return: embeddings: each concept is converted into a list of int, correspending to one concept 
    :type: List[np.array]  # Unless we use pytorch for parallel computation 

    Problem solving in process
    - You need to convert the embedding algo TY wrote for your purposes here, notably a diff input 
    - We will call this function once for each knowledge base 
    '''
    
    all_embeddings = [] 
    for knowledge in knowledges:
        statements = [entry["statement"] for entry in knowledge[:100]]
        print(f'first 10 statements is {statements[:10]}, the len of statements being {len(statements)}')
        raw_output = vo.embed(  
            statements,  # Expecting List[str]
            model="voyage-3-large",
            output_dimension=EMB_DIM,  # at least 256. Do we want to pass on this argument?
        )
        print(f'the raw output is {raw_output}') # With an intention ot check out whether raw output contains nan or not. 
        cur_emb = raw_output.embeddings#[:,1:]  # NEP I am concerned that this might be a nested list, causing errors.
        #EMB_DIM = cur_emb.shape[1]
        print(f'the 1st item of embedding looks like {cur_emb[:1]}, with the len of {len(cur_emb)}, and type of {type(cur_emb)}')

        if np.isnan(cur_emb).any():
            print(f"Found NaN in embedding. Replacing NaN values.")
            cur_emb = np.nan_to_num(cur_emb, nan=0.0)  # Replace NaNs with 0.0

        # Processing exceptions 
        if len(cur_emb) != len(knowledge[:100]) or \
            np.isnan(cur_emb).any() or \
            np.isinf(cur_emb).any():
            raise ValueError("Failed to embed strings. Invalid embeddings returned.")
        cur_emb = np.array(cur_emb) # Converting [List[List[int]] to np.array
        print(f'current type of cur_emb is {cur_emb}')
        all_embeddings.append(cur_emb)

    return all_embeddings  # This actually gives us List[List[List[int]]]

# Alternative embedding w/ OpenAI 


# Clustering (K-means for MVP)
def cluster_kmeans(embeddings: List[np.array]) -> List[np.array]:

    '''
    params etc 
    :param embeddings: all knowledge items converted to sentence embeddings.
    :type embeddings: List[np.array]

    :return: list_labels, each is a array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :type: List[nd.array] (num_turns, (num_items,))

    Tianyi: there should be improved kmean for self-emerged clusters.
    '''
    # Silhouette to decide the best k 
    data = np.vstack(embeddings) # Vertically stacks arrays, shape will be [len(list)*num_items, embed_dims]

    best_k = 3 # We give it a default (to avoid the None case; also k is a decent guess)
    best_score = -1
    for k in range(2,11):
        kmean_for_best_k = KMeans(n_clusters=k, random_state=42).fit(data)
        score = silhouette_score(data, kmean_for_best_k.labels_)
        if score > best_score:
            best_score = score
            best_k = k


    # K-means to do the clustering 
    list_labels = []
    for embedding in embeddings:
        kmeans = KMeans(n_clusters=best_k, random_state=42)  # For reproductivity 
        labels_kmeans = kmeans.fit_predict(embedding)
        list_labels.append(labels_kmeans)  # In visualization you should associate each np.array (labels contained in one KB) a KB identifier  
    return list_labels 


# UMAP --> Apply PCA and leave out the essential dimensions (2 PC; initial and final comparison)
def dim_red(embeddings: List[np.array])-> List[np.array]: 
    
    '''
    :param: embeddings: a numpy array that each row corresponds to one item and each column is one dim of that item
    :type embeddings: np.array (num_items, dims)

    :return all_reduced_embeddings, 2-dim primary dims after umap
    :rtype: np.array [num_turns * num_items, prim_dims], num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 
    '''

    # Initialize UMAP
    reduced_embeddings = [] 
    for embedding in embeddings:
        print(f"Processing embedding with shape: {embedding.shape}")
        if embedding.shape[0] <= 2:
            raise ValueError(f"Too few rows in embedding for dimensionality reduction: {embedding.shape[0]}")
        
        # Adjust n_neighbors based on data size
        n_neighbors = min(embedding.shape[0] - 1, 15)
        reducer = umap.UMAP(n_neighbors = n_neighbors, 
                            min_dist = 0.1, # default
                            n_components=2,   # We want to draw a 2D graph in the end 
                            random_state = 42

        )
        print(f"Using n_neighbors={n_neighbors} for embedding with {embedding.shape[0]} rows.")

        reduced_embedding = reducer.fit_transform(embedding) 
        reduced_embeddings.append(reduced_embedding)
        all_reduced_embeddings = np.vstack(reduced_embeddings)
    return all_reduced_embeddings

# Visualization
def visualization(prim_dims: np.array, list_labels: List[np.array]):
    '''
    :params prim_dims: the primary dims after applying dim reduction. 
    :type prim_dims: [num_turns * num_items, prim_dims]

    :return: list_labels, each is a array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]. len(list)=600, each np.array[100,]
    :rtype: List[nd.array] (num_turns, (num_items,))

    '''
    # Visualize cluster in 2D

    all_labels = np.array(list_labels).flatten() # We want to acquire all unique cluster values to make the graph.
    print(f'all_labels shape {all_labels.shape}')
    # identifying data to be highlighted (e.g., here we are trying to highlight first and last 1%, correponding to the first and last 6 knowledge_bases)
    array_length = len(list_labels[0]) # each knowledge base is of the same length 
    portion_to_identify = int(len(list_labels)*(1/600)) # We only identify the first and last KB; but this can change
    source_ids_start, source_ids_end = np.full(array_length * portion_to_identify, 1), np.full(array_length * portion_to_identify, 2)
    source_ids = np.concatenate([source_ids_start, np.full(array_length *(len(list_labels)-portion_to_identify*2),-1), source_ids_end])
    print(f'prim_dims[:,0] shape {prim_dims[:,0].shape}')
    print(f'source_ids shape {source_ids.shape}')

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=prim_dims[:,0],
        y=prim_dims[:,1],
        hue = all_labels, # all the unique cluster values can be found here 
        style = np.where(source_ids == -1, "Unlabeled", source_ids), # data point will not be highlighted if their source_id == -1
        palette= 'Set1',
        legend='full'
    )
    plt.title("Clusters Visualized in 2D")
    plt.xlabel("Principal Dimension 1")
    plt.ylabel("Principal Dimension 2")
    plt.legend(title='Cluster', loc='best')

    # Make timestamped directory for this experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Use the timestamp to record running data files 
    backup_dir = f"/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/analysis/run-{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Directory created: {os.path.exists(backup_dir)}")

    plt.savefig(f"{backup_dir}/clusters_2d.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures

    # Visualize cluster trends over time (We only create one graph all all turns)
        
    # a) Create a DataFrame containing turns and assigned clusters
    records = []
    for turn_idx, label_array in enumerate(list_labels):
        print(f'label_array is {label_array}, with the shape of {label_array.shape}')
        for lab in label_array:
            records.append({"turn": turn_idx, "cluster": lab}) # Effectively a list of dict (turns*len_knowledge_base,)
    df = pd.DataFrame(records)

    # b) Count how many items are in each cluster per turn
    cluster_counts = df.groupby(['turn', 'cluster']).size().reset_index(name='num_items') 
    # Groups all rows by the unique (turn, cluster) pairs in df, resulting in .size(), which is a Series where the index is (turn, cluster) and the value is the count of items.effectively one group=one point on the graph=times of that cluster at given turn

    # c) Pivot so each cluster becomes a column, indexed by turn
    pivot_df = cluster_counts.pivot(index='turn', columns='cluster', values='num_items').fillna(0)  

    # d) Plot the line chart (cluster sizes vs. turn)
    pivot_df.plot(kind='line', marker='o', figsize=(8, 6))
    plt.title("Cluster Size Over Turns")
    plt.xlabel("Turns")
    plt.ylabel("Number of Items in Each Cluster")
    plt.legend(title='Cluster', loc='best')

    os.makedirs(backup_dir, exist_ok=True)
    plt.savefig("{backup_dir}/cluster_trends.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures


if __name__ == "__main__":
    import logging
    import voyageai
    voyageai.api_key = "pa-D7cexR9gsRuYWYv3IfAS9h-aIV_bKjuTJ9nx7n59Du8"
    print("Current working directory:", os.getcwd())

    # Gather the data (maybe merging every KB  into a bigger .json)
    all_data = [] 
    folder_path = "/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-230124/round000"  # We need to pass on this from the terminal with one line that defines the folder path.
    if not os.path.exists("/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-230124/round000"):
        raise FileNotFoundError(f'Folder Path {folder_path} does not exist.')

    file_names = sorted([x for x in os.listdir(folder_path) if x.endswith(".json") and x.startswith("knowledge-turn")], 
                        key=lambda x: int(x.split('.')[0].replace("knowledge-turn", "")))

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            kb = json.load(file)
            all_data.append(kb)
    if not all_data:
        print("No knowledge base data is found, Exiting.")
        exit()

    if not all(isinstance(kb, list) and all(check_type(item, Dict[str, Union[int, str]]) for item in kb) for kb in all_data):
        raise ValueError("Data formate invalid: all_data should be a list of lists of strings")

    logging.info(f"Loaded {len(all_data)} knowledge bases.")
    # NEP I guess this would log/print?

    # Define an instance of voyageai client 
    vo = voyageai.Client()

    # Get embeddings
    try:
        embeddings = embedding(vo, all_data)  # List[np.array]
        logging.info("Embeddings generated successfully.")
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        exit(1) # What does this "1" do?


    # Clustering w/ kmeans 
    try:
        clusters = cluster_kmeans(embeddings)
        logging.info("Clustering completed.")
    except Exception as e:
        logging.error(f"Error during clustering: {e}")
        exit(1)

    # Reducing dimensions for visualization 
    try:
        reduced_embeddings = dim_red(embeddings)  # List[np.array]
        logging.info("Dimensionality reduction completed.")
    except Exception as e:
        logging.error(f"Error during dimensionality reduction: {e}")
        exit(1)

    # Visualization
    try:
        visualization(reduced_embeddings, clusters)
        logging.info("Visualization completed. PDFs generated.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        exit(1)



