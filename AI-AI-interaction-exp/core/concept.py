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
import numpy as np
import pandas as pd 
import json
import os
import voyageai
import voyageai.client  # NEP I don't know the function of this one. 
from typing import List, Dict, Tuple, Mapping
from typeguard import check_type

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # Clustering algos 
import umap
import matplotlib.pyplot as plt 
import seaborn as sns
import OpenAI from "openai";

EMB_DIM = 256


# Gather the data (maybe merging every KB  into a bigger .json)
all_data = [] 
folder_path = "AI-AI-interaction-exp/data/runs/run-20241231-232135"  # We need to pass on this from the terminal with one line that defines the folder path.

file_names = sorted([x for x in os.listdir(folder_path) if x.endswith(".json")], 
                    key=lambda x: int(x.split('.')[0].replace("knowledge-turn", "")))

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        kb = json.load(file)
        
        all_data.append(kb)


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
        cur_emb = vo.embed(
            knowledge,  
            model="voyage-3-large",
            output_dimension=EMB_DIM,  # at least 256. Do we want to pass on this argument?
        ).embeddings


        # Processing exceptions 
        if not isinstance(cur_emb, list) or \
            len(cur_emb) != len(knowledge) or \
            np.isnan(cur_emb).any() or \
            np.isinf(cur_emb).any():
            raise ValueError("Failed to embed strings. Invalid embeddings returned.")
        all_embeddings.append(cur_emb)

    return all_embeddings

# Alternative embedding w/ OpenAI 


# UMAP --> Apply PCA and leave out the essential dimensions (2 PC; initial and final comparison)
def dim_red(embeddings: List[np.array])-> List[np.array]: 
    
    '''
    :param: embeddings: a numpy array that each row corresponds to one item and each column is one dim of that item
    :type embeddings: np.array (num_items, dims)

    :return reduce_dembeddings, 2-dim primary dims after umap
    :rtype: List[np.array] (num_turns, (num_items, prim_dims)), num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 
    '''

    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors = 15, # default 
                        min_dist = 0.1, # default
                        n_components=2,   # We want to draw a 2D graph in the end 
                        random_state = 42

    )
    reduced_embeddings = [] 
    for embedding in embeddings:
        reduced_embedding = reducer.fit_transform(embedding) 
        reduced_embeddings.append(reduced_embedding)

    return reduced_embeddings


# Clustering (K-means for MVP)
def cluster_kmeans(reduced_embeddings: List[np.array]) -> List[np.array]:

    '''
    params etc 

    :return: labels_kmeans, each is a array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :type: List[nd.array] (num_turns, (num_items,))

    Tianyi: there should be improved kmean for self-emerged clusters.
    '''
    list_labels = []
    for reduced_embedding in reduced_embeddings:
        kmeans = KMeans(n_clusters=3, random_state=42)  # For reproductivity 
        labels_kmeans = kmeans.fit_predict(reduced_embedding)
        list_labels.append(labels_kmeans)
    return list_labels 

# Visualization
def visualization(prim_dims: List[np.array], list_labels: List[np.array]):
    '''
    :params pri_dims: the primary dims after applying dim reduction. 
    :type final_arrays: (num_turns, (num_items, prim_dims)

    :return: list_labels, each is a array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype: List[nd.array] (num_turns, (num_items,))


    '''
    # Visualize clusters in 2D (We create a cluster for each turn)
    for turn, prim_dim in enumerate(prim_dims):
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x=prim_dim[:,0],
            y=prim_dim[:,1],
            hue = list_labels[turn], # names of the columns
            palette= 'Set1',
            legend='full'
        )
        plt.title("Clusters Visualized in 2D")
        plt.xlabel("Principal Dimension 1")
        plt.ylabel("Principal Dimension 2")
        plt.legend(title='Cluster', loc='best')
        plt.savefig(f"turn_{turn}_clusters_2d.pdf", format="pdf")

    # Visualize cluster trends over time (We only create one graph all all turns)
        
    # a) Create a DataFrame containing turns and assigned clusters
    records = []
    for turn_idx, label_array in enumerate(list_labels):
        for lab in label_array:
            records.append({"turn": turn_idx, "cluster": lab}) # Effectively a list of dict (turns*len_knowledge_base,)

    df = pd.DataFrame(records)

    # b) Count how many items are in each cluster per turn
    cluster_counts = df.groupby(['turn', 'cluster']).size().reset_index(name='num_items') # Groups all rows by the unique (turn, cluster) pairs in df, resulting in .size(), which is a Series where the index is (turn, cluster) and the value is the count of items.effectively one group=one point on the graph=times of that cluster at given turn

    # c) Pivot so each cluster becomes a column, indexed by turn
    pivot_df = cluster_counts.pivot(index='turn', columns='cluster', values='num_items').fillna(0)  

    # d) Plot the line chart (cluster sizes vs. turn)
    pivot_df.plot(kind='line', marker='o', figsize=(8, 6))
    plt.title("Cluster Size Over Turns")
    plt.xlabel("Turns")
    plt.ylabel("Number of Items in Each Cluster")
    plt.legend(title='Cluster', loc='best')
    plt.savefig("cluster_size_trends.pdf", format="pdf")



