## Data Panel 1: Time periods

Each row (aka entry) in this data panel represents **one model family (3.5 / 4) x one time period (0 - 130)**.

Metadata:

```
Data columns (total 12 columns):                                                                                                                                                              
 #   Column                                         Non-Null Count  Dtype                                                                                                                     
---  ------                                         --------------  -----                                                                                   
 0   gpt_version                                    212 non-null    int64
 1   overall_nsamples                               212 non-null    int64
 2   overall_mean_turns                             212 non-null    float64
 3   overall_mean_conversation_length               212 non-null    object
 4   overall_mean_prompt_length                     212 non-null    float64
 5   concept_diversity                              211 non-null    float64
 6   concept_diversity_user_concepts_explicit       210 non-null    float64
 7   concept_diversity_assistant_concepts_explicit  210 non-null    float64
 8   concept_diversity_user_concepts_related        210 non-null    float64
 9   concept_diversity_assistant_concepts_related   211 non-null    float64
 10  concept_diversity_user                         210 non-null    float64
 11  concept_diversity_assistant                    211 non-null    float64                                                                                                                   
dtypes: float64(9), int64(2), object(1)                                                                                                                        
```

Summary statistics:

```
Panel temporal_panel1 summary stats:                                                                                                                                                          
       gpt_version  overall_nsamples  overall_mean_turns  \                                                                                                                                   
count   212.000000        212.000000          212.000000
mean      0.933962       3952.806604            4.570882
std       0.656991       2530.934695            1.138373
min       0.000000          8.000000            2.766276
25%       0.750000       2175.000000            3.614331   
50%       1.000000       3062.500000            4.325795   
75%       1.000000       5550.250000            5.428680   
max       2.000000      12011.000000            8.521739   

       overall_mean_prompt_length  concept_diversity  \
count                  212.000000         211.000000   
mean                  2537.125728          13.533190   
std                   1737.809097           1.134250   
min                    220.250000          11.001848   
25%                   1291.280582          13.015447   
50%                   2161.543513          14.135161   
75%                   3140.807373          14.431996   
max                  14052.853991          14.846604   

       concept_diversity_user_concepts_explicit  \
count                                210.000000    
mean                                  13.752535    
std                                    1.048348    
min                                    9.055444    
25%                                   13.162018    
50%                                   14.056204    
75%                                   14.586082    
max                                   15.421219    

       concept_diversity_assistant_concepts_explicit  \
count                                     210.000000   
mean                                       13.821369   
std                                         1.003113   
min                                        11.645738   
25%                                        13.379817   
50%                                        14.267582   
75%                                        14.600491   
max                                        15.067221   

       concept_diversity_user_concepts_related  \
count                               210.000000    
mean                                 12.670080    
std                                   1.479132    
min                                   9.533953    
25%                                  11.664979    
50%                                  13.532186    
75%                                  13.860079    
max                                  14.973264


       concept_diversity_assistant_concepts_related  concept_diversity_user  \
count                                    211.000000              210.000000   
mean                                      13.001632               13.115732   
std                                        1.408472                1.335378   
min                                       10.185058               10.334172   
25%                                       12.172062               12.332491   
50%                                       13.835092               13.814929   
75%                                       14.139105               14.219296   
max                                       14.981592               14.835354   

       concept_diversity_assistant  
count                   211.000000  
mean                     13.516868  
std                       1.182802  
min                      11.001848  
25%                      12.949381  
50%                      14.213409  
75%                      14.416157  
max                      14.928843
```

## Data Panel 2: Time periods x Concepts

Each row (aka entry) in this data panel represents **one model family (3.5 / 4) x one time period (0 - 130) x one concept**. It contains the statistics about conversations that contain the concept during the time period.

Metadata:

```
Data columns (total 8 columns):
 #   Column                                        Non-Null Count   Dtype  
---  ------                                        --------------   -----  
 0   cluster_nsamples_user_concepts_explicit       494984 non-null  int64  
 1   cluster_nsamples_assistant_concepts_explicit  494984 non-null  int64  
 2   cluster_nsamples_user_concepts_related        494984 non-null  int64  
 3   cluster_nsamples_assistant_concepts_related   494984 non-null  int64  
 4   cluster_nsamples                              494984 non-null  int64  
 5   cluster_mean_turns                            494984 non-null  float64
 6   cluster_mean_conversation_length              494984 non-null  float64
 7   cluster_mean_prompt_length                    494984 non-null  float64
dtypes: float64(3), int64(5)
```

Summary statistics:

```
Panel temporal_panel2 summary stats:
       cluster_nsamples_user_concepts_explicit  \
count                            494984.000000    
mean                                107.731418    
std                                 188.583634    
min                                   0.000000    
25%                                   0.000000    
50%                                   1.000000    
75%                                 143.000000    
max                                1536.000000    

       cluster_nsamples_assistant_concepts_explicit  \
count                                 494984.000000   
mean                                     350.705336   
std                                      608.953425   
min                                        0.000000   
25%                                        1.000000   
50%                                        2.000000   
75%                                      453.000000   
max                                     4704.000000   

       cluster_nsamples_user_concepts_related  \
count                           494984.000000    
mean                               259.021492    
std                                486.347805    
min                                  0.000000    
25%                                  0.000000    
50%                                  1.000000    
75%                                252.000000    
max                               2442.000000    

       cluster_nsamples_assistant_concepts_related  cluster_nsamples  \
count                                494984.000000     494984.000000   
mean                                    277.564727        601.664615   
std                                     498.871444       1055.989390   
min                                       0.000000          1.000000   
25%                                       0.000000          1.000000   
50%                                       1.000000          2.000000   
75%                                     314.000000        743.000000   
max                                    2904.000000       7125.000000   

       cluster_mean_turns  cluster_mean_conversation_length  \
count       494984.000000                     494984.000000   
mean             4.282059                       6096.137729   
std              4.183929                       6211.071162   
min              2.000000                         11.000000   
25%              2.000000                       3591.000000   
50%              3.121328                       5597.575994   
75%              4.880977                       6714.159836   
max            292.000000                     310128.000000


       cluster_mean_prompt_length  
count               494984.000000  
mean                  2494.835432  
std                   4640.205353  
min                      0.000000  
25%                    398.000000  
50%                   2135.812949  
75%                   3621.669683  
max                 298850.000000
```


## Data Panel 3: Users

Each row (aka entry) in this data panel represents **one user**. It contains the statistics about the user's conversations.

Metadata:

```
Data columns (total 18 columns):                                                                                                                                                              
 #   Column                                         Non-Null Count  Dtype                                                                                                                     
---  ------                                         --------------  -----                                                                                                                     
 0   language                                       27215 non-null  object                                                                                                                    
 1   location                                       27215 non-null  object                                                                                                                    
 2   nsamples                                       27215 non-null  int64                                                                                                                     
 3   nsamples_temporal_composition                  27215 non-null  object                                                                                                                    
 4   nsamples_version_composition                   27215 non-null  object                                                                                                                    
 5   temporal_extension                             27215 non-null  float64                                                                                                                   
 6   version_diversity                              27215 non-null  float64                                                                                                                   
 7   mean_turns                                     27215 non-null  float64                                                                                                                   
 8   mean_conversation_length                       27215 non-null  float64                                                                                                                   
 9   mean_prompt_length                             27215 non-null  float64                                                                                                                   
 10  concept_diversity                              22423 non-null  float64                                                                                                                   
 11  concept_diversity_user_concepts_explicit       9693 non-null   float64                                                                                                                   
 12  concept_diversity_assistant_concepts_explicit  19402 non-null  float64                                                                                                                   
 13  concept_diversity_user_concepts_related        13534 non-null  float64                                                                                                                   
 14  concept_diversity_assistant_concepts_related   15573 non-null  float64                                                                                                                   
 15  concept_diversity_user_across_time             27215 non-null  object                                                                                                                    
 16  concept_diversity_assistant_across_time        27215 non-null  object                                                                                                                    
 17  concept_diversity_across_time                  27215 non-null  object                                                                                                                    
dtypes: float64(10), int64(1), object(7)                                  
```


Summary statistics:

```
Panel user_panel summary stats:                                                                                                                                                               
           nsamples  temporal_extension  version_diversity    mean_turns  \                                                                                                                   
count  27215.000000        27215.000000       27215.000000  27215.000000                                                                                                                      
mean      19.679478            4.195894           0.894515      4.668817                                                                                                                      
std      108.409565            7.963516           0.185029      3.319881                                                                                                                      
min        4.000000            0.000000           0.200000      0.800000                                                                                                                      
25%        5.000000            0.000000           0.825702      2.250000                                                                                                                      
50%        7.000000            0.500000           1.000000      3.714286                                                                                                                      
75%       13.000000            4.150276           1.000000      5.818182                                                                                                                      
max     6526.000000           58.522966           1.000000     59.200000                                                                                                                      
                                                                                                                                                                                              
       mean_conversation_length  mean_prompt_length  concept_diversity  \                                                                                                                     
count              27215.000000        27215.000000       22423.000000                                                                                                                        
mean                4961.046670         1823.686603          13.155989                                                                                                                        
std                 5646.259629         4410.326727           2.649140                                                                                                                        
min                   15.500000            0.000000          -4.830075                                                                                                                        
25%                 2101.416667          180.614583          11.417278                                                                                                                        
50%                 3964.000000          618.333333          13.685109                                                                                                                        
75%                 6163.214286         2754.150000          15.026740                                                                                                                        
max               179299.250000       171766.777778          16.609640                                                                                                                        
                                                                                                                                                                                              
       concept_diversity_user_concepts_explicit  \                                                                                                                                            
count                               9693.000000                                                                                                                                               
mean                                  12.626040                                                                                                                                               
std                                    3.891723                                                                                                                                               
min                                   -6.642448                                                                                                                                               
25%                                   11.006079                                                                                                                                               
50%                                   13.021219                                                                                                                                               
75%                                   15.407087                                                                                                                                               
max                                   16.609640                                                                                                                                               
                                                                                                                                                                                              
       concept_diversity_assistant_concepts_explicit  \                                                                                                                                       
count                                   19402.000000                                                                                                                                          
mean                                       13.067709                                                                                                                                          
std                                         3.182969                                                                                                                                          
min                                        -6.415037   
25%                                        11.615350   
50%                                        13.685109   
75%                                        15.026740   
max                                        16.609640   

       concept_diversity_user_concepts_related  \
count                             13534.000000    
mean                                 11.522020    
std                                   3.569836    
min                                  -4.830075    
25%                                   9.473724    
50%                                  11.546811    
75%                                  14.351820    
max                                  16.609640    

       concept_diversity_assistant_concepts_related
count                                  15573.000000  
mean                                      12.050327  
std                                        3.404171  
min                                       -4.830075  
25%                                       10.006013  
50%                                       12.901350  
75%                                       14.689280  
max                                       16.609640
```