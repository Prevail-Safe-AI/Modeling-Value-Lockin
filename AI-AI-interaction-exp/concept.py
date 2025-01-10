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
from numpy.linalg import norm 

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Adjust the import search path to include its parent and parent-parent folders
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path


EMB_DIM = 256



# Embedding: str --> vector representations for later nummerical processing 
def embedding(vo: voyageai.Client, knowledges: List[List[Dict[str, Union[int, str]]]]) -> List[np.array]: 
    '''
    :param knowledges: non-trivial concepts extracted from knowledge base
    :type: List[]

    :return: embeddings: each concept is converted into a list of int, correspending to one concept 
    :type: List[np.array]  # Unless we use pytorch for parallel computation 

    Problem solving in process
    - You need to convert the embedding algo TY wrote for your purposes here, notably a diff input 
    - We will call this function once for each knowledge base 
    '''
    
    all_embeddings = [] 
    all_mappings = [] # creating a list of mappings out of {statement:embedding}
    for idx, knowledge in enumerate(knowledges):
        statements = [entry["statement"] for entry in knowledge[:100]]
        print(f'first 10 statements is {statements[:10]}, the len of statements being {len(statements)}')
        raw_output = vo.embed(  
            statements,  # Expecting List[str]
            model="voyage-3-large",
            output_dimension=EMB_DIM,  # at least 256. Do we want to pass on this argument?
        )
        print(f'the raw output is {raw_output}') # With an intention ot check out whether raw output contains nan or not. 
        cur_emb = raw_output.embeddings#[:,1:]  #[nup.items, num_dims]
        #EMB_DIM = cur_emb.shape[1]
        print(f'the 1st item of embedding looks like {cur_emb[:1]}, with the len of {len(cur_emb)}, and type of {type(cur_emb)}')

        # To np.array and then normalize 
        cur_emb = np.array(cur_emb)  # Converting [List[List[int]] to np.array]
        cur_emb = StandardScaler().fit_transform(cur_emb)  # Normalize embeddings
        print(f"In the {idx} round of embedding, Mean: {np.mean(cur_emb)}, Std: {np.std(cur_emb)}")

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

    return all_embeddings, all_mappings  # List[np.array]; List[Dict[str, float]]

# Alternative embedding w/ OpenAI 


# Clustering (K-means for MVP)
def cluster_kmeans(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> List[np.array]:

    '''
    :param embeddings: all knowledge items converted to sentence embeddings.
    :type embeddings: List[np.array]

    :param knowledges: all knowledge items passed on from the simulation 
    :type knwledges: List[List[Dict[str, Union[int, str]]]], where each item is a dict of id and statement. 

    :return list_labels: each is a array of an index of cluster that examples belongs to, like [1,2,3,1,2,3,2]
    :rtype list_labels: List[nd.array] (num_turns, (num_items,))

    :return all_statements_to_labels: mappings from statement str to clustering labsl 
    :rtype all_statements_to_labels: List[Dict[str, int]]
    '''
    # Silhouette to decide the best k 
    data = np.vstack(embeddings) # Vertically stacks arrays, shape will be [len(list)*num_items, embed_dims]

    best_k = 3 # We give it a default (to avoid the None case; also 3 is a decent guess)
    best_score = -1
    for k in range(2,100):
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
    list_labels = []
    all_statements_to_labels = []

    for embedding, (idx,knowledge) in zip(embeddings, enumerate(knowledges)):
        print(f"In the {idx} round of cluster_kmeans, Mean: {np.mean(embedding)}, Std: {np.std(embedding)}")
        kmeans = KMeans(n_clusters=best_k, random_state=42)  # For reproductivity 
        labels_kmeans = kmeans.fit_predict(embedding)
        list_labels.append(labels_kmeans)  # In visualization you should associate each np.array (labels contained in one KB) a KB identifier  

        # Calculate and log the Silhouette Score for the final K-means clustering
        final_score = silhouette_score(embedding, kmeans.labels_)
        print(f"Final Silhouette Score for K={best_k}: {final_score}")


        # Statement-to-label
        statement_to_label = {(idx, entry["statement"]): label for idx, (entry, label) in enumerate(zip(knowledge[:100], labels_kmeans))}
        all_statements_to_labels.append(statement_to_label)

        for tuple_pair in statement_to_label:
            print(f'tuple_pair: {tuple_pair}, Value:{statement_to_label[tuple_pair]}')

    return list_labels, all_statements_to_labels

'''
# UMAP --> Apply UMAP and leave out the essential dimensions (2 PC; initial and final comparison)
def dim_red(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]])-> List[np.array]: 
    
    
    :param: embeddings: a numpy array that each row corresponds to one item and each column is one dim of that item
    :type embeddings: np.array (num_items, dims)

    :return all_reduced_embeddings, 2-dim primary dims after umap
    :rtype: np.array [num_turns * num_items, prim_dims], num_turns refers to turns in user-tutor chat, corresponding to the num of knowledge-base, too; num_items refers to num of knowledge items on one knowledge base 


    # Initialize UMAP
    reduced_embeddings = [] 
    all_reduced_mappings = []
    for embedding, knowledge in zip(embeddings, knowledges):
        print(f"Processing embedding with shape: {embedding.shape}")
        if embedding.shape[0] <= 2:
            raise ValueError(f"Too few rows in embedding for dimensionality reduction: {embedding.shape[0]}")
        
        # Adjust n_neighbors based on data size
        #n_neighbors = min(embedding.shape[0] - 1, 15)
        n_neighbors = 40
        reducer = umap.UMAP(n_neighbors = n_neighbors, 
                            min_dist = 0.1, # default
                            n_components=2,   # We want to draw a 2D graph in the end 
                            random_state = 42

        )
        print(f"Using n_neighbors={n_neighbors} for embedding with {embedding.shape[0]} rows.")

        reduced_embedding = reducer.fit_transform(embedding) 
        reduced_embeddings.append(reduced_embedding)
        all_reduced_embeddings = np.vstack(reduced_embeddings)

        # Build a dict for statement:embedding mappings 
        know_reduced_map = {entry["statement"]: reduced for entry, reduced in zip(knowledge[:100], reduced_embeddings)}  # You may retrieve the embedding with their corresponding statement as the key.
        all_reduced_mappings.append(know_reduced_map)

    return all_reduced_embeddings, all_reduced_mappings

'''

# t-sne for dimension reduction 
def dim_red_tsne(embeddings: List[np.array], knowledges: List[List[Dict[str, Union[int, str]]]]) -> List[np.array]:
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
    reduced_embeddings = []
    all_reduced_mappings = []

    for embedding, knowledge in zip(embeddings, knowledges):
        print(f"Processing embedding with shape: {embedding.shape}")

        if embedding.shape[0] <= 2:
            raise ValueError(f"Too few rows in embedding for dimensionality reduction: {embedding.shape[0]}")

        # Adjust perplexity based on data size
        perplexity = min(embedding.shape[0] - 1, 90)  # Default max perplexity for t-SNE

        reducer = TSNE(
            n_components=2,   # We want to draw a 2D graph in the end
            perplexity=perplexity,
            random_state=42,
            init='pca',  # t-SNE's random initialization
            learning_rate='auto'
        )

        print(f"Using perplexity={perplexity} for embedding with {embedding.shape[0]} rows.")

        reduced_embedding = reducer.fit_transform(embedding)
        reduced_embeddings.append(reduced_embedding)
        all_reduced_embeddings = np.vstack(reduced_embeddings)

        # Build a dict for statement:embedding mappings
        know_reduced_map = {
            entry["statement"]: reduced for entry, reduced in zip(knowledge[:100], reduced_embedding)
        }  # You may retrieve the embedding with their corresponding statement as the key.
        all_reduced_mappings.append(know_reduced_map)

    return all_reduced_embeddings, all_reduced_mappings

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
    print(f'Number of all unique clusters {np.unique(all_labels)}')
    # identifying data to be highlighted (e.g., here we are trying to highlight first and last 1%, correponding to the first and last 6 knowledge_bases)
    array_length = len(list_labels[0]) # each knowledge base is of the same length 
    portion_to_identify = int(len(list_labels)*(1/600)) # We only identify the first and last KB; but this can change
    source_ids_start, source_ids_end = np.full(array_length * portion_to_identify, 1), np.full(array_length * portion_to_identify, 2)
    source_ids = np.concatenate([source_ids_start, np.full(array_length *(len(list_labels)-portion_to_identify*2),-1), source_ids_end])

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=prim_dims[:,0],  # all should be one-dim (x, y, hue, style)
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
        for lab in label_array:  # One lab is one label of that embedding 
            records.append({"turn": turn_idx, "cluster": lab}) # Effectively a list of dict (turns*len_knowledge_base,)
    df = pd.DataFrame(records)

    # b) Count how many items are in each cluster per turn
    cluster_counts = df.groupby(['turn', 'cluster']).size().reset_index(name='num_items') 
    # Groups all rows by the unique (turn, cluster) pairs in df, resulting in .size(), 
    # which is a Series where the index is (turn, cluster) and the value is the count of items. 
    # Effectively one group=one point on the graph=times of that cluster at given turn
    # Check unique clusters and how many counters per turn 
    print("Unique clusters in DataFrame:", df['cluster'].unique())
    print("Cluster counts per turn:\n", cluster_counts)

    # c) Pivot so each cluster becomes a column, indexed by turn
    pivot_df = cluster_counts.pivot(index='turn', columns='cluster', values='num_items').fillna(0)  
    # check whether all clusters are represented
    print("Pivoted DataFrame (cluster sizes over turns):\n", pivot_df)

    # d) Plot the line chart (cluster sizes vs. turn)
    pivot_df.plot(kind='line', marker='o', figsize=(8, 6))
    plt.title("Cluster Size Over Turns")
    plt.xlabel("Turns")
    plt.ylabel("Number of Items in Each Cluster")
    plt.legend(title='Cluster', loc='best')

    os.makedirs(backup_dir, exist_ok=True)
    plt.savefig(f"{backup_dir}/cluster_trends.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures


if __name__ == "__main__":
    import logging
    import voyageai
    voyageai.api_key = "pa-D7cexR9gsRuYWYv3IfAS9h-aIV_bKjuTJ9nx7n59Du8"
    print("Current working directory:", os.getcwd())

    # Gather the data (maybe merging every KB  into a bigger .json)
    all_data = [] 
    # real set (100 turns)
    #folder_path = "/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-232135/round000"  # We need to pass on this from the terminal with one line that defines the folder path.
    #if not os.path.exists("/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-232135/round000"):
    #    raise FileNotFoundError(f'Folder Path {folder_path} does not exist.')

    # toy set (6turns)
    folder_path = "/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-230124/round000"  # We need to pass on this from the terminal with one line that defines the folder path.
    if not os.path.exists("/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/runs/run-20241231-230124/round000"):
        raise FileNotFoundError(f'Folder Path {folder_path} does not exist.')

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
    # NEP I guess this would log/print?

    # Define an instance of voyageai client 
    vo = voyageai.Client()

    # Get embeddings
    try:
        embeddings, all_mappings = embedding(vo, all_data)  # List[np.array]
        logging.info("Embeddings generated successfully.")
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        exit(1) # What does this "1" do?

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
    data = np.vstack(embeddings)
    print(f'the data type is {type(data)}')
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend for headless servers

    # Adjust perplexity based on data size
    perplexity = min(data.shape[0] - 1, 90)  # Default max perplexity for t-SNE
    reducer = TSNE(
        n_components=2,   # We want to draw a 2D graph in the end
        perplexity=perplexity,
        random_state=42,
        init='pca',  # t-SNE's random initialization
        learning_rate='auto'
    )

    reduced_embeddings = reducer.fit_transform(data)  # 'data' is the stacked embeddings
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    plt.title("Visualizing Embeddings Before Clustering")
    # Make timestamped directory for this experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Use the timestamp to record running data files 
    backup_dir = f"/home/ubuntu/experimentation-fs/zhonghao/Modeling-Value-Lockin/AI-AI-interaction-exp/data/analysis/run-{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Directory created: {os.path.exists(backup_dir)}")

    plt.savefig(f"{backup_dir}/natural_cluster.pdf", format="pdf")
    plt.close()  # Close the plot to avoid overlapping figures



    # Clustering w/ kmeans 
    print(f"Shape Checks before K-means")
    print("Length of embeddings:", len(embeddings))
    print("Shapes of each embedding array:", [e.shape for e in embeddings])
    print("Length of knowledges:", len(all_data))
    print("Lengths of each knowledge list:", [len(k) for k in all_data])
    try:
        clusters, statements_to_labels = cluster_kmeans(embeddings, all_data)
        logging.info("Clustering completed.")
    except Exception as e:
        logging.error(f"Error during clustering: {e}")
        exit(1)

    # Retrieve two labels 
    label1 = statements_to_labels[5][(1, 'Information literacy crucial for navigating vast digital information.')]
    label2 = statements_to_labels[5][(0, 'Apply basic first aid to treat wounds and stabilize injuries.')]
    label3 = statements_to_labels[5][(16, 'Basic first aid: clean, apply antibiotic ointment, cover injuries.')]

    print(f"After the kmeans, label1 is {label1}, label2 is {label2}, and label3 is {label3}")

    # Reducing dimensions for visualization 
    try:
        reduced_embeddings, all_reduced_mappings = dim_red_tsne(embeddings, all_data)  # List[np.array]
        logging.info("Dimensionality reduction completed.")
    except Exception as e:
        logging.error(f"Error during dimensionality reduction: {e}")
        exit(1)

    # Retrieve two reduced embeddings 
    #reduced_v1 = all_reduced_mappings[5]["Energy conservation: total energy remains constant, forms change."]
    #reduced_v2 = all_reduced_mappings[5]["Quantum fluctuations challenge conservation of energy principles slightly"]        
    # Calculating their Euclidean distances. 
    #reduced_euclidean_distance = norm(reduced_v1 - reduced_v2)
    #print("Euclidean Distance after Dim Red between two identified items:", reduced_euclidean_distance)

    # Visualization
    try:
        visualization(reduced_embeddings, clusters)
        logging.info("Visualization completed. PDFs generated.")
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        exit(1)



