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
import logging

from typing import List, Dict, Tuple, Mapping, Union
from typeguard import check_type

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # Clustering algos 
import hdbscan
from sklearn.metrics import silhouette_score # to decide best k in k-means 
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt 
import seaborn as sns
import voyageai
from numpy.linalg import norm 


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


# Adjust the import search path to include its parent and parent-parent folders
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

EMB_DIM = 256
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

from kbutils.json_utils import load_file, dump_file


# Gather the data (maybe merging every KB  into a bigger .json)
all_data = [] 
# real set (100 turns)

# icl-1024 turns 
folder_path = "./data/runs/ICL-run3-20250125-151456/round000"  # We need to pass on this from the terminal with one line that defines the folder path.
content_save_path = "./data/runs/ICL-run3-20250125-151456"
if not os.path.exists("./data/runs/ICL-run3-20250125-151456/round000"):
    raise FileNotFoundError(f'Folder Path {folder_path} does not exist.')


print("Current working directory:", os.getcwd())



file_names = sorted([x for x in os.listdir(folder_path) if x.endswith(".json") and x.startswith("knowledge-turn")], 
                    key=lambda x: int(x.split('.')[0].replace("knowledge-turn", "")))

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        kb = json.load(file)
        all_data.append(kb[:100])
if not all_data:
    print("No knowledge base data is found, Exiting.")
    exit()

if not all(isinstance(kb, list) and all(check_type(item, Dict[str, Union[int, str]]) for item in kb) for kb in all_data):
    raise ValueError("Data formate invalid: all_data should be a list of lists of strings")

logging.info(f"Loaded {len(all_data)} knowledge bases.")

# Define an instance of voyageai client 
vo = voyageai.Client()



# Embedding: str --> vector representations for later nummerical processing 
def embedding(vo: voyageai.Client, knowledges: List[List[Dict[str, Union[int, str]]]]) -> List[np.array]: 
    '''
    :param knowledges: non-trivial concepts extracted from knowledge base
    :type knowledges: List[]

    :return: embeddings: each concept is converted into a list of int, correspending to one concept 
    :rtype: List[np.array]  # Unless we use pytorch for parallel computation 

    Problem solving in process
    - You need to convert the embedding algo wrote for your purposes here, notably a diff input 
    - We will call this function once for each knowledge base 
    '''
    
    all_embeddings = [] 
    all_mappings = [] # creating a list of mappings out of {statement:embedding}
    for idx, knowledge in enumerate(knowledges):
        statements = [entry["statement"] for entry in knowledge[:100]]
        #print(f'first 10 statements is {statements[:10]}, the len of statements being {len(statements)}')
        raw_output = vo.embed(  
            statements,  # Expecting List[str]
            model="voyage-3-large",
            output_dimension=EMB_DIM,  # at least 256. Do we want to pass on this argument?
        )
        #print(f'the raw output is {raw_output}') # With an intention ot check out whether raw output contains nan or not. 
        cur_emb = raw_output.embeddings#[:,1:]  #[nup.items, num_dims]
        #EMB_DIM = cur_emb.shape[1]
        # print(f'the 1st item of embedding looks like {cur_emb[:1]}, with the len of {len(cur_emb)}, and type of {type(cur_emb)}')

        # To np.array and then normalize 
        # Apply L2 normalization (row-wise normalization for embeddings)
        # cur_emb = normalize(cur_emb, norm='l2', axis=1)

        # Sanity check: Each row should have unit norm
        norms = np.linalg.norm(cur_emb, axis=1)
        #print("Norms after L2 normalization:", norms)  # Should print 1.0 for all rows

        if np.isnan(cur_emb).any():
            print(f"Found NaN in embedding. Replacing NaN values.")
            cur_emb = np.nan_to_num(cur_emb, nan=0.0)  # Replace NaNs with 0.0

        # Processing exceptions 
        if len(cur_emb) != len(knowledge[:100]) or \
            np.isnan(cur_emb).any() or \
            np.isinf(cur_emb).any():
            raise ValueError("Failed to embed strings. Invalid embeddings returned.")
 
        all_embeddings.append(cur_emb)

        # Build a dict for statement:embedding mappings 
        know_embed_map = {key: embedding for key, embedding in zip(statements, cur_emb)}  # You may retrieve the embedding with their corresponding statement as the key.
        all_mappings.append(know_embed_map)

    if isinstance(all_embeddings, list):
        all_embeddings = [np.array(embedding) for embedding in all_embeddings]
    return all_embeddings, all_mappings  # List[np.array]; List[Dict[str, float]]

# Alternative embedding w/ OpenAI 

# Calculating the pair-wise Euclidean distances of all embedding in a given knowledge base.
def pairwise_dis(all_embeddings: List[np.array])-> List[float]:
    '''
    :param all_embeddings: all embeddings from all KBs
    :type all_embeddings: List[np.array]

    :return all_distances: each is all the pair-distance among items of a given KB
    :rtype List[float]

    '''
    all_distances = [] 
    for idx, embeddings in enumerate(all_embeddings):
        distances = pdist(embeddings, metric='euclidean')
        average_distance = np.mean(distances)
        print(f"Average Distance in the {idx} round is {average_distance}")
        all_distances.append(average_distance)
    return all_distances 


# Clustering (K-means for MVP)
def cluster_kmeans(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> np.array:

    '''
    :param embeddings: all knowledge items converted to sentence embeddings.
    :type embeddings: List[np.array]

    :param knowledges: all knowledge items passed on from the simulation 
    :type knwledges: List[List[Dict[str, Union[int, str]]]], where each item is a dict of id and statement. 

    :return list_labels: an array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype list_labels: np.array (num_turns * num_items,)

    :return all_statements_to_labels: mappings from statement str to clustering labsl 
    :rtype all_statements_to_labels: List[Dict[str, int]]
    '''
    dump_file(embeddings, f"{content_save_path}/concept-embeddings.json")
    dump_file(knowledges, f"{content_save_path}/concept-knowledge.json")
    
    # Silhouette to decide the best k 
    length_kbs = len(knowledges)
    data = np.vstack(embeddings) # Vertically stacks arrays, shape will be [len(list)*num_items, embed_dims]

    best_k = 3 # We give it a default (to avoid the None case; also 3 is a decent guess)
    best_score = -1
    for k in range(2,20):
        kmean_for_best_k = KMeans(n_clusters=k, random_state=42).fit(data)
        score = silhouette_score(data, kmean_for_best_k.labels_)
        print(f"Current k is {k}, and current silhouette score {score}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f'Current Best K is {best_k}')
    if best_k > 8:
        best_k = 8
    # K-means to do the clustering 

    # Clustering 
    kmeans = KMeans(n_clusters=best_k, random_state=42)  # For reproductivity 
    labels_kmeans = kmeans.fit_predict(data)


    # Calculate and log the Silhouette Score for the final K-means clustering
    final_score = silhouette_score(data, kmeans.labels_)
    print(f"Final Silhouette Score for K={best_k}: {final_score}")

    # to list
    list_labels = labels_kmeans.reshape(length_kbs,-1).tolist() 

    all_statements_to_labels = []
    # Statement-to-label
    for knowledge, labels in zip(knowledges, list_labels):
        statement_to_label = {(idx, entry["statement"]): label for idx, (entry, label) in enumerate(zip(knowledge[:100], labels))}
        all_statements_to_labels.append(statement_to_label)

    for tuple_pair in statement_to_label:
        print(f'tuple_pair: {tuple_pair}, Value:{statement_to_label[tuple_pair]}')

    return labels_kmeans, all_statements_to_labels

def cluster_hdbscan(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> np.array:
    '''
    :param embeddings: all knowledge items converted to sentence embeddings.
    :type embeddings: List[np.array]

    :param knowledges: all knowledge items passed on from the simulation 
    :type knwledges: List[List[Dict[str, Union[int, str]]]], where each item is a dict of id and statement. 

    :return list_labels: an array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype list_labels: np.array (num_turns * num_items,)

    :return all_statements_to_labels: mappings from statement str to clustering labels 
    :rtype all_statements_to_labels: List[Dict[str, int]]
    '''
    dump_file(embeddings, f"{content_save_path}/concept-embeddings.json")
    dump_file(knowledges, f"{content_save_path}/concept-knowledge.json")
    
    # Combine all embeddings into a single array
    length_kbs = len(knowledges)
    data = np.vstack(embeddings)  # Shape: [len(list)*num_items, embed_dims]

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=2, metric='euclidean')
    labels_hdbscan = clusterer.fit_predict(data)

    # Check for noise points
    noise_count = np.sum(labels_hdbscan == -1)
    print(f"Number of noise points: {noise_count}")

    all_statements_to_labels = {}
    # Map statements to labels
    all_knowledges = [item for knowledge in knowledges for item in knowledge[:100]] # stack the whole list in one single list

    for idx, (entry, label) in enumerate(zip(all_knowledges, labels_hdbscan)):
        all_statements_to_labels[(idx, entry["statement"])] = label

    for tuple_pair in all_statements_to_labels:
        print(f'tuple_pair: {tuple_pair}, Value: {all_statements_to_labels[tuple_pair]}')

    return labels_hdbscan, all_statements_to_labels

# clustering algorithm that runs much faster: https://github.com/TutteInstitute/evoc
def cluster_evoc(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> np.array:
    '''
    :param embeddings: all knowledge items converted to sentence embeddings.
    :type embeddings: List[np.array]

    :param knowledges: all knowledge items passed on from the simulation 
    :type knwledges: List[List[Dict[str, Union[int, str]]]], where each item is a dict of id and statement. 

    :return list_labels: an array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype list_labels: np.array (num_turns * num_items,)

    :return all_statements_to_labels: mappings from statement str to clustering labels 
    :rtype all_statements_to_labels: List[Dict[str, int]]
    '''
    dump_file(embeddings, f"{content_save_path}/concept-embeddings.json")
    dump_file(knowledges, f"{content_save_path}/concept-knowledge.json")
    
    # Combine all embeddings into a single array
    length_kbs = len(knowledges)
    data = np.vstack(embeddings)  # Shape: [len(list)*num_items, embed_dims]
    
    import evoc

    print(f"Clustering (multithreading)... (current time: {time.strftime('%Y%m%d-%H%M%S')})")
    clusterer = evoc.EVoC(
        base_min_cluster_size = 2,
        n_epochs=300,
        n_neighbors=512,
        node_embedding_dim=64,
        next_cluster_size_quantile=0.8,
        noise_level=0,
    )
    cluster_labels = clusterer.fit_predict(data)
    cluster_layers = clusterer.cluster_layers_
    hierarchy = clusterer.cluster_tree_
    print(f"Clustering complete. (current time: {time.strftime('%Y%m%d-%H%M%S')})")
    
    # Identify parent clusters of each string and each cluster
    layer_counts = [len(set(layer) - set([-1])) for layer in cluster_layers] + [1]
    parent_child_pairs = set()
    print(f"Layer counts: {layer_counts} ({len(layer_counts)} layers)")
    
    # Back up cluster_layers
    print("Backing up cluster_layers...")
    dump_file(cluster_layers, f"{content_save_path}/cluster_layers.json")
    
    # Use the highest level with count >20 as labels_evoc
    use_layer = 0
    for layer_idx, layer_count in enumerate(layer_counts):
        if layer_count > 2:
            use_layer = layer_idx
    
    labels_evoc = cluster_layers[use_layer]

    # Check for noise points
    noise_count = np.sum(labels_evoc == -1)
    print(f"Number of noise points: {noise_count}")

    all_statements_to_labels = {}
    # Map statements to labels
    all_knowledges = [item for knowledge in knowledges for item in knowledge[:100]] # stack the whole list in one single list

    for idx, (entry, label) in enumerate(zip(all_knowledges, labels_evoc)):
        all_statements_to_labels[(idx, str(entry["statement"]))] = int(label)

    for tuple_pair in all_statements_to_labels:
        print(f'tuple_pair: {tuple_pair}, Value: {all_statements_to_labels[tuple_pair]}')
    
    dump_file(labels_evoc, f"{content_save_path}/concept-labels_evoc.json")
    dump_file(list(all_statements_to_labels.items()), f"{content_save_path}/concept-all_statements_to_labels.json")

    return labels_evoc, all_statements_to_labels


# UMAP --> Apply UMAP and leave out the essential dimensions (2 PC; initial and final comparison)
def dim_red_umap(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]])-> List[np.array]: 
    
    '''
    :param: embeddings: a numpy array that each row corresponds to one item and each column is one dim of that item
    :type embeddings: np.array (num_items, dims)

    :return all_reduced_embeddings, 2-dim primary dims after umap
    :rtype: np.array [num_turns * num_items, prim_dims], num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 
    '''

    # Initialize UMAP
    all_embeddings = np.vstack(embeddings)
    all_reduced_mappings = []
    print(f"Processing embedding with shape: {all_embeddings.shape}")
    if all_embeddings.shape[0] <= 2:
        raise ValueError(f"Too few rows in embedding for dimensionality reduction: {all_embeddings.shape[0]}")
    
    # Adjust n_neighbors based on data size
    n_neighbors = min(all_embeddings.shape[0] - 1, 200)
    reducer = umap.UMAP(n_neighbors = n_neighbors, 
                        min_dist = 0.1, # default
                        n_components=2,   # We want to draw a 2D graph in the end 
                        random_state = 42

    )
    print(f"Using n_neighbors={n_neighbors} for embedding with {all_embeddings.shape[0]} rows.")

    all_reduced_embeddings = reducer.fit_transform(all_embeddings) 

    # Build a dict for statement:embedding mappings 
    all_knowledges = [item for knowledge in knowledges for item in knowledge[:100]] # stack the whole list in one single list
    all_reduced_mappings = {entry["statement"]: reduced for entry, reduced in zip(all_knowledges, all_reduced_embeddings)}  # You may retrieve the embedding with their corresponding statement as the key.

    return all_reduced_embeddings, all_reduced_mappings


# t-sne for dimension reduction 
def dim_red_tsne(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> np.array:
    '''
    :param embeddings: a numpy array where each row corresponds to one item, and each column is one dimension of that item
    :type embeddings: List[np.array] (num_items, dims)

    :param knowledges: a list of knowledge bases corresponding to the embeddings
    :type knowledges: List[List[Dict[str, Union[int, str]]]]

    :return all_reduced_embeddings: 2D primary dimensions after t-SNE
    :rtype: np.array [num_turns * num_items, prim_dims], num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 
    
    :return all_reduced_mappings: list of dicts mapping statements to reduced embeddings
    :rtype: List[Dict[str, np.array]]
    '''

    # Initialize t-SNE
    all_embeddings = np.vstack(embeddings)
    print(f"Processing embedding with shape: {all_embeddings.shape}")

    if all_embeddings.shape[0] <= 2:
        raise ValueError(f"Too few rows in embedding for dimensionality reduction: {all_embeddings.shape[0]}")

    # Adjust perplexity based on data size
    perplexity = min(all_embeddings.shape[0] - 1, 50)  # Default max perplexity for t-SNE

    reducer = TSNE(
        n_components=2,   # We want to draw a 2D graph in the end
        perplexity=perplexity,
        random_state=42,
        init='pca',  # t-SNE's random initialization
        learning_rate='auto'
    )

    print(f"Using perplexity={perplexity} for embedding with {all_embeddings.shape[0]} rows.")

    reduced_embedding = reducer.fit_transform(all_embeddings)

    
    # Build a dict for statement:embedding mappings 
    all_knowledges = [item for knowledge in knowledges for item in knowledge[:100]] # stack the whole list in one single list
    all_reduced_mappings = {entry["statement"]: reduced for entry, reduced in zip(all_knowledges, reduced_embeddings)}  # You may retrieve the embedding with their corresponding statement as the key.
  
    return reduced_embedding, all_reduced_mappings

def dim_red(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> List[np.array]:
    '''
    :param: embeddings: a numpy array that each row corresponds to one item and each column is one dim of that item
    :type embeddings: np.array (num_items, dims)

    :return all_reduced_embeddings, 2-dim primary dims after PCA
    :rtype: np.array [num_turns * num_items, prim_dims], num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 
    '''

    # Stack all embeddings into a single array
    all_embeddings = np.vstack(embeddings)
    all_reduced_mappings = []
    print(f"Processing embedding with shape: {all_embeddings.shape}")
    if all_embeddings.shape[0] <= 2:
        raise ValueError(f"Too few rows in embedding for dimensionality reduction: {all_embeddings.shape[0]}")

    # Initialize PCA with 2 components
    pca = PCA(n_components=3, random_state=42)
    print(f"Performing PCA for embedding with {all_embeddings.shape[0]} rows.")

    # Fit and transform embeddings using PCA
    all_reduced_embeddings = pca.fit_transform(all_embeddings)

    # Build a dict for statement:embedding mappings 
    all_knowledges = [item for knowledge in knowledges for item in knowledge[:100]]  # stack the whole list in one single list
    all_reduced_mappings = {entry["statement"]: reduced for entry, reduced in zip(all_knowledges, all_reduced_embeddings)}  # You may retrieve the embedding with their corresponding statement as the key.

    return all_reduced_embeddings, all_reduced_mappings


# Visualization
def visualization(prim_dims: np.array, all_distances: List[float], labels_kmeans: np.array):
    '''
    :param prim_dims: the primary dims after applying dim reduction. 
    :type prim_dims: [num_turns * num_items, prim_dims]

    # param all_distances: all the pair-wise Euclidean distances, each float per KB
    # type all_distances: List[float]

    :return labels_kmeans: an array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype labels_kmeans: np.array (num_turns *num_items,)

    '''
    # Visualize cluster in 2D
    unique_labels = np.unique(labels_kmeans)
    print(f'Number of all unique clusters {unique_labels}')

    # identifying data to be highlighted (e.g., here we are trying to highlight first and last 1%, correponding to the first and last 6 knowledge_bases)
    #array_length = len(list_labels[0]) # each knowledge base is of the same length 
    #portion_to_identify = int(len(list_labels)*(1/600)) # We only identify the first and last KB; but this can change
    #source_ids_start, source_ids_end = np.full(array_length * portion_to_identify, 1), np.full(array_length * portion_to_identify, 2)
    #source_ids = np.concatenate([source_ids_start, np.full(array_length *(len(list_labels)-portion_to_identify*2),-1), source_ids_end])

    plt.figure(figsize=(8,6))
    print(f"Before the cluster plot, data shape A is {prim_dims.shape}, and a sample is {prim_dims[0]}")
    sns.scatterplot(
        x=prim_dims[:,0],  # all should be one-dim (x, y, hue, style)
        y=prim_dims[:,1],
        hue = labels_kmeans, # Cluster label for each data point 
        #style = np.where(source_ids == -1, "Unlabeled", source_ids), # data point will not be highlighted if their source_id == -1
        palette= 'Set1',
        legend='full',
    )
    plt.title("Clusters Visualized in 2D")
    plt.xlabel("Principal Dimension 1")
    plt.ylabel("Principal Dimension 2")
    plt.legend(title='Cluster', loc='best')

    # Make timestamped directory for this experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    plt.savefig(f"{content_save_path}/clusters_2d.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures

    # Visualize cluster trends over time (We only create one graph all all turns)
        
    # a) Create a DataFrame containing turns and assigned clusters
    records = []
    list_labels = labels_kmeans.reshape(-1,100)
    print(f"Currently we have {len(list_labels)} lists, and each contain 100 labels")
    for turn_idx, label_array in enumerate(list_labels):
        print(f'label_array is {label_array}, with the shape of {label_array.shape}')
        for lab in label_array:  # One lab is one label of that embedding 
            records.append({"turn": turn_idx, "cluster": lab}) # Effectively a list of dict (turns*len_knowledge_base,)
    df = pd.DataFrame(records)

    # b) Count how many items are in each cluster per turn
    cluster_counts = df.groupby(['turn', 'cluster']).size().reset_index(name='num_items') 
    # Groups all rows by the unique (turn, cluster) pairs in df, resulting in .size() which is a Series where the index is (turn, cluster) and the value is the count of items. Effectively one group=one point on the graph=times of that cluster at given turn. Check unique clusters and how many counters per turn 
    print("Unique clusters in DataFrame:", df['cluster'].unique())
    print("Cluster counts per turn:\n", cluster_counts)

    # c) Pivot so each cluster becomes a column, indexed by turn
    pivot_df = cluster_counts.pivot(index='turn', columns='cluster', values='num_items').fillna(0)  
    # check whether all clusters are represented
    print("Pivoted DataFrame (cluster sizes over turns):\n", pivot_df)

    # d) Plot the line chart (cluster sizes vs. turn)
    pivot_df.plot(kind='line', marker="", figsize=(8, 6))
    plt.title("Cluster Size Over Turns")
    plt.xlabel("Turns")
    plt.ylabel("Number of Items in Each Cluster")
    plt.legend(title='Cluster', loc='best')

    os.makedirs(content_save_path, exist_ok=True)
    plt.savefig(f"{content_save_path}/cluster_trends.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures

    # Visualize a pair-wise Euclidean distances trends 

    print(f"Right before visualization, distances are {all_distances[:10]}, and its shape is {len(all_distances)}")
    x_values = range(len(all_distances))
    
    plt.figure(figsize=(8, 5))  # Set the figure size
    plt.plot(x_values,
            all_distances, 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            markersize=3)

    # axises 
    plt.xlabel("Knowledge Base Index", fontsize=12)
    plt.ylabel("Average Euclidean Distances", fontsize=12)
    plt.title("Average Distance Trend", fontsize=14)

    # Add grid for better readability
    plt.grid(True)
    # Use the timestamp to record running data files 
    os.makedirs(content_save_path, exist_ok=True)
    print(f"Directory created: {os.path.exists(content_save_path)}")

    plt.savefig(f"{content_save_path}/euclidean_distance.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures

    
    '''
    # Visualize a heatmap of 3*3 Euclidean Distances 

    # items to be specified,3*3 from statement_to_embeddings 
    item_embeddings = [all_mappings[6]["Data quality complexity arises from industry, context, and purposes interdependencies."],
             all_mappings[6]["Data quality dimensions include accuracy, completeness, consistency, and validity."],
             all_mappings[6]["Data quality issues arise from multiple sources & lifecycle stages."],
             all_mappings[3]["Humans learn through iterative discovery, reflection, and consistent practice."],
             all_mappings[3]["Human learning occurs through iterative cycles of discovery, reflection, practice."],
             all_mappings[3]["Humans learn through combination of discovery, reflection, and practice."],
             all_mappings[3]["Energy is conserved, it can't be created or destroyed."],
             all_mappings[3]["Energy conservation drives natural processes towards equilibrium."],
             all_mappings[3]["Energy can transform between forms, yet total energy remains constant."]
             ] 
    item_labels = ["data1",
                   "data2",
                   "data3",
                   "human1",
                   "human2",
                   "human3",
                   "energy1",
                   "energy2",
                   "energy3"]
    # Ecludian distances intra all 9 pairs, in a np.array
    dist_matrix = pairwise_distances(item_embeddings, metric="euclidean")

    # Symmetric and zero-diagonal 
    for i in range(9):
        for j in range(i+1,9):
            dist_matrix[j,i] = dist_matrix[i,j]
        dist_matrix[i,i] = 0.0

    plt.figure(figsize=(8,6))
    sns.heatmap(
        dist_matrix,
        xticklabels=item_labels,
        yticklabels=item_labels,
        cmap="Blues",
        annot=True,  # show numeric values,
        fmt=".2f" # round to 2 decimals 
    ) 

    plt.title("Pairwise Distances Between Embeddings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Use the timestamp to record running data files 
    os.makedirs(content_save_path, exist_ok=True)
    print(f"Directory created: {os.path.exists(content_save_path)}")

    plt.savefig(f"{content_save_path}/heatmap.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures
    '''

if __name__ == "__main__":

    # Get embeddings
    try:
        embeddings = load_file(f"{content_save_path}/concept-embeddings.json")
        embeddings = [np.array(embedding) for embedding in embeddings]
    except:
        print("Generating embeddings...")
        embeddings, _ = embedding(vo, all_data)  # List[np.array]
        logging.info("Embeddings generated successfully.")

    '''
    # Calculate the embeddings for some known items  
    # Retrieve two embeddings whose corresponding statements are known to us.
    v1 = all_mappings[5]["Information literacy crucial for navigating vast digital information."]
    v2 = all_mappings[5]["Apply basic first aid to treat wounds and stabilize injuries."]
    v3 = all_mappings[5]["Basic first aid: clean, apply antibiotic ointment, cover injuries."]
    # Calculating their Euclidean distances. 
    euclidean_distance_diff_items = norm(v1 - v2)
    print("Euclidean Distance between two identified items:", euclidean_distance_diff_items)
    euclidean_distance_simi_items = norm(v2 - v3)
    print("Euclidean Distance between two identified items:", euclidean_distance_simi_items)

    # Visualize embeddings before clustering 
    import numpy as np
    data = embeddings[5] # We only run the last KB this time
    # print(f'the data type is {type(data)}')
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend for headless servers

    reduced_embeddings = []
    for embedding in embeddings:
        # Adjust perplexity based on data size
        perplexity = min(embedding.shape[0] - 1, 90)  # Default max perplexity for t-SNE
        reducer = TSNE(
            n_components=2,   # We want to draw a 2D graph in the end
            perplexity=perplexity,
            random_state=42,
            init='pca',  # t-SNE's random initialization
            learning_rate='auto'
        )
        reduced_embedding = reducer.fit_transform(embedding)  # 'data' is the stacked embeddings
        reduced_embeddings.append(reduced_embedding)
    reduced_embeddings = np.vstack(reduced_embeddings)
    print(f"Before the cluster plot, data shape B is {reduced_embeddings.shape}, \n and a sample is {reduced_embeddings[0]}")
    #sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], palette= 'Set1', legend='full')

    perplexity = min(data.shape[0] - 1, 90)  # Default max perplexity for t-SNE
    print("current perplexity is ", perplexity)
    reducer = TSNE(
        n_components=2,   # We want to draw a 2D graph in the end
        perplexity=perplexity,
        random_state=42,
        init='pca',  # t-SNE's random initialization
        learning_rate='auto'
    )
    reduced_embeddings = reducer.fit_transform(data)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title("Visualizing Embeddings Before Clustering")
    # Make timestamped directory for this experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")

  
    plt.savefig(f"{content_save_path}/natural_cluster.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures

    '''

    # Clustering w/ kmeans 
    print(f"Shape Checks before K-means")
    print("Length of embeddings:", len(embeddings))
    print("Shapes of each embedding array:", embeddings[0].shape)
    print("Length of knowledges:", len(all_data))
    print("Lengths of each knowledge list:", [len(k) for k in all_data[:5]])
    clusters, statements_to_labels = cluster_evoc(embeddings, all_data)
    logging.info("Clustering completed.")

    '''
    # Retrieve two labels 
    label1 = statements_to_labels[5][(1, 'Information literacy crucial for navigating vast digital information.')]
    label2 = statements_to_labels[5][(0, 'Apply basic first aid to treat wounds and stabilize injuries.')]
    label3 = statements_to_labels[5][(16, 'Basic first aid: clean, apply antibiotic ointment, cover injuries.')]

    print(f"After the kmeans, label1 is {label1}, label2 is {label2}, and label3 is {label3}")
    '''
    # Reducing dimensions for visualization 
    reduced_embeddings, _ = dim_red(embeddings, all_data)  # List[np.array]
    logging.info("Dimensionality reduction completed.")

    '''
    # Retrieve two reduced embeddings 
    #reduced_v1 = all_reduced_mappings[5]["Energy conservation: total energy remains constant, forms change."]
    #reduced_v2 = all_reduced_mappings[5]["Quantum fluctuations challenge conservation of energy principles slightly"]        
    # Calculating their Euclidean distances. 
    #reduced_euclidean_distance = norm(reduced_v1 - reduced_v2)
    #print("Euclidean Distance after Dim Red between two identified items:", reduced_euclidean_distance)
    '''

    # Pairwise Euclidean Distances 
    all_distances = pairwise_dis(embeddings)
    logging.info("All Euclidean Distances Calculated.")

    # Visualization
    visualization(reduced_embeddings, all_distances, clusters)
    logging.info("Visualization completed. PDFs generated.")



