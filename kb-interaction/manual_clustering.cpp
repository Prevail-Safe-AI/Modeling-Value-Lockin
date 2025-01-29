#include "kbutils/json.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <utility>
#include <cstring>
#include <cassert>
#include <fstream>
#include <map>
#include <set>
using namespace std;
using json = nlohmann::json;

bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

// Computes:
//   1) Longest Common Substring length
//   2) Max total length of up to 2 non-overlapping common substrings
//   3) Max total length of up to 3 non-overlapping common substrings
//
// All in O(n*m) time.
tuple<int,int,int> analyzeCommonSubstrings(const string &s1, const string &s2) {
    int n = (int)s1.size();
    int m = (int)s2.size();

    // c[i][j] = length of the longest common suffix of
    //           s1[0..i-1] and s2[0..j-1] that ends exactly at i-1, j-1.
    // If s1[i-1] == s2[j-1], then c[i][j] = 1 + c[i-1][j-1], else 0.
    vector<vector<int>> c(n+1, vector<int>(m+1, 0));

    // Compute the c array.
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= m; j++){
            if(s1[i-1] == s2[j-1]){
                c[i][j] = c[i-1][j-1] + 1;
            } else {
                c[i][j] = 0;
            }
        }
    }

    // 1) The length of the single longest common substring
    //    is simply max of c[i][j].
    int longestSingle = 0;
    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= m; j++){
            longestSingle = max(longestSingle, c[i][j]);
        }
    }

    // dp[i][j][k] = maximum total length of up to k non-overlapping
    // common substrings using s1 up to index i-1 and s2 up to index j-1.
    static const int KMAX = 3; // we'll compute up to k=3
    vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(m+1, vector<int>(KMAX+1, 0)));

    // Fill dp in increasing order of i,j
    for(int i = 0; i <= n; i++){
        for(int j = 0; j <= m; j++){
            for(int k = 0; k <= KMAX; k++){
                if(i > 0) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k]);
                }
                if(j > 0) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k]);
                }

                // If k > 0 and we have a common suffix of length L here:
                if(k > 0 && i > 0 && j > 0 && c[i][j] > 0) {
                    int L = c[i][j];
                    // We can take that substring of length L and add to dp[i-L][j-L][k-1]
                    int iPrev = i - L;
                    int jPrev = j - L;
                    // Make sure indices are valid
                    if(iPrev >= 0 && jPrev >= 0) {
                        dp[i][j][k] = max(dp[i][j][k], dp[iPrev][jPrev][k-1] + L);
                    }
                }
            }
        }
    }

    // dp[n][m][1] is the maximum sum length of up to 1 substring,
    // which should match longestSingle. But we'll trust our direct
    // c-array maximum for part (1). You could verify they match.

    int bestForTwo   = dp[n][m][2]; // sum length for up to 2 substrings
    int bestForThree = dp[n][m][3]; // sum length for up to 3 substrings

    return make_tuple(longestSingle, bestForTwo, bestForThree);
}

int getWeight(const string &s1, const string &s2) {
    auto [len1, len2, len3] = analyzeCommonSubstrings(s1, s2);
    return len1 + len2 + len3;
}

namespace DSU{
    const int MAX_NODES = 500000;
    int fa[MAX_NODES], component_reductions;

    void init(){
        component_reductions = 0;
        for (int i=0; i<MAX_NODES; ++i) 
            fa[i] = i;
    }

    int ancestor(int x){
        return (fa[x] == x) ? x : (fa[x] = ancestor(fa[x]));
    }

    void unite(int x, int y){
        assert(x < MAX_NODES);
        assert(y < MAX_NODES);
        x = ancestor(x);
        y = ancestor(y);
        if (x != y) ++component_reductions;
        fa[x] = y;
    }

    int num_components(int num_nodes){
        return num_nodes - component_reductions;
    }
}

const int ITEMS_PER_TIMESTEP = 100, MAX_STR_LEN = 1000;

int getNodeID(int timestep, int index) {
    assert(index < ITEMS_PER_TIMESTEP);
    assert(index >= 0);
    return ITEMS_PER_TIMESTEP * timestep + index;
}

vector<pair<int,int>> sorted_edges[MAX_STR_LEN + 5];
vector<vector<string>> snapshots, snapshots_raw;

void saveSortedEdges() {
    json data = json::array();
    for(int i=0; i<=MAX_STR_LEN; ++i) {
        json current_level = json::array();
        if (!sorted_edges[i].empty()) {
            cout << "Level " << i << ": " << sorted_edges[i].size() << " edges" << endl;
        }
        for (auto p : sorted_edges[i]) {
            current_level.push_back(p.first);
            current_level.push_back(p.second);
        }
        data.push_back(current_level);
    }
    string path = "data/analysis/cpp-cluster/sorted_edges.json";
    ofstream o(path);
    o << setw(4) << data << endl;
}

void loadSortedEdges() {
    string path = "data/analysis/cpp-cluster/sorted_edges.json";
    ifstream f(path);
    json data;
    f >> data;
    int edge_counts = 0;
    for (int i=0; i<=MAX_STR_LEN; ++i) {
        int sz = (int)data[i].size();
        assert(sz % 2 == 0);
        for (int j=0; j<sz; j+=2) {
            sorted_edges[i].push_back(make_pair(data[i][j], data[i][j+1]));
            edge_counts += 1;
        }
    }
    cout << "Loaded " << edge_counts << " edges." << endl;
}

void testEdge(int timestep_x, int index_x, int timestep_y, int index_y) {
    int weight = getWeight(snapshots[timestep_x][index_x], snapshots[timestep_y][index_y]);
    if (weight > MAX_STR_LEN) {
        cout << "Anomaly: " << weight << endl;
        weight = MAX_STR_LEN;
    }
    sorted_edges[weight].push_back(make_pair(getNodeID(timestep_x, index_x), getNodeID(timestep_y, index_y)));
}

const int WEIGHT_CUTOFF = 81;
const int CROSS_SNAPSHOT_WEIGHT_CUTOFF = 81;

void saveClusters(int stamp, bool aligned=false, bool renumber=false) {
    json data = json::array();
    map<int,int> renumber_map;

    for (int i=0; i<(int)snapshots.size(); ++i) {
        json current_snapshot = json::array();
        for (int j=0; j<ITEMS_PER_TIMESTEP; ++j) {
            int label = DSU::ancestor(getNodeID(i, j));

            if (renumber) {
                if (renumber_map.find(label) == renumber_map.end()) {
                    renumber_map[label] = (int)renumber_map.size();
                }
                label = renumber_map[label];
            }

            json current_item = {
                {"id", j},
                {"statement_cleaned", snapshots[i][j]},
                {"statement_raw", snapshots_raw[i][j]},
                {"label", label}
            };
            current_snapshot.push_back(current_item);
        }
        json current_object = {
            {"snapshot_id", i},
            {"content", current_snapshot}
        };
        data.push_back(current_object);
    }

    string path;
    if (!aligned)
        path = "data/analysis/cpp-cluster/" + to_string(stamp) + ".json";
    else
        path = "data/analysis/cpp-cluster/" + to_string(stamp) + "-" + to_string(CROSS_SNAPSHOT_WEIGHT_CUTOFF) + "-aligned.json";
    
    ofstream o(path);
    o << setw(4) << data << endl;
}

int highest_set_bit(int v){
    int c = 0;
    while(v) {
        v >>= 1;
        c += 1;
    }
    return c;
}

string simplify(const string &s) {
    vector<string> stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am", "will", "would", "may", "might", "have", "has", "had", "do", "does", "did", "doing", "done", "to", "in", "on", "at", "by", "with", "for", "of", "about", "as", "into", "onto", "upon", "under", "over", "through", "between", "among", "within", "without", "before", "after", "since", "until", "while", "throughout", "upon", "up", "down", "out", "off", "away", "back", "around", "about", "above", "below", "beneath", "beside", "between", "beyond", "inside", "outside", "underneath", "under", "over", "across", "along", "against", "behind", "before", "beneath", "beside", "between", "beyond", "by", "down", "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "since", "through", "throughout", "to", "toward", "under", "underneath", "until", "up", "upon", "with", "within", "without", "and", "or", "but", "so", "yet", "for", "nor", "after", "although", "as", "because", "before", "if", "since", "though", "unless", "until", "when", "where", "while", "that", "which", "who", "whom", "whose", "whichever", "whoever", "whomever", "what", "whatever", "when", "whenever", "where", "wherever", "why", "how", "however", "whence", "whither", "whence", "whither", "there", "here", "now", "then", "thus", "hence", "accordingly", "consequently", "therefore", "moreover", "furthermore", "however", "nevertheless", "nonetheless", "otherwise", "instead", "meanwhile", "likewise", "similarly", "indeed", "certainly", "sure", "perhaps", "maybe", "possibly", "probably", "certainly", "definitely", "absolutely", "always", "never", "sometimes", "often", "usually", "generally", "mostly", "partly", "partially", "completely", "fully", "entirely", "wholly", "altogether", "absolutely", "totally", "perfectly", "imperfectly", "partially", "partly", "mostly", "mostly", "completely", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    };
    set<string> stopwords_set(stopwords.begin(), stopwords.end());

    vector<string> words;
    string current_word;
    for (char c : s) {
        if (isalpha(c)) {
            current_word += tolower(c);
        }
        else if ('0' <= c && c <= '9') {
            current_word += c;
        }
        else {
            if (current_word.size() > 1) {
                if (stopwords_set.find(current_word) == stopwords_set.end()) {
                    words.push_back(current_word);
                }
            }
            current_word.clear();
        }
    }
    if (current_word.size() > 1) {
        if (stopwords_set.find(current_word) == stopwords_set.end()) {
            words.push_back(current_word);
        }
    }

    string result = " ";
    for (string word : words) {
        result += word + " ";
    }
    return result;
}

void save_snapshots() {
    json data = json::array();
    for (int i=0; i<(int)snapshots.size(); ++i) {
        json current_snapshot = json::array();
        for (int j=0; j<ITEMS_PER_TIMESTEP; ++j) {
            json current_item = {
                {"id", j},
                {"statement_cleaned", snapshots[i][j]},
                {"statement_raw", snapshots_raw[i][j]}
            };
            current_snapshot.push_back(current_item);
        }
        json current_object = {
            {"snapshot_id", i},
            {"content", current_snapshot}
        };
        data.push_back(current_object);
    }

    string path = "data/analysis/cpp-cluster/snapshots.json";
    ofstream o(path);
    o << setw(4) << data << endl;
}

void feature_test() {
    json tmp_array = json::array();
    string statement = "Hi!";
    json element = {
        {"id", 10},
        {"statement", statement},
        {"label", 5}
    };
    json full = {
        {"outer_id", 0},
        {"content", element}
    };
    tmp_array.push_back(full);
    cout << tmp_array << endl;
    
    string simplify_test_str = "Alice the quick brown fox jumps over the lazy dog, tripping on a rock for the 1e9th time.";
    cout << simplify(simplify_test_str) << endl;
}

int main(){
    feature_test();
    DSU::init();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector < string > directories = {
        "data/runs/run-20250125-001430-ICL/round000",
        "data/runs/run-20250125-125731-ICL-continuation1/round000",
        "data/runs/run-20250126-043940-ICL-continuation2/round000",
        "data/runs/run-20250126-165145-ICL-continuation3/round000",
        // "data/runs/ICL-run3-20250125-151456/round000"
    };

    for (auto directory : directories) {
        for (int turn=1; ; ++turn) {
            string turn_num = to_string(turn);
            if (turn < 10) {
                turn_num = "0" + turn_num;
            }
            
            string path = directory + "/knowledge-turn" + turn_num + ".json";
            if (!fileExists(path)) {
                cout << "Finished reading " << directory << " at turn " << turn << endl;
                break;
            }

            ifstream f(path);
            json data = json::parse(f);

            snapshots.emplace_back(ITEMS_PER_TIMESTEP);
            snapshots_raw.emplace_back(ITEMS_PER_TIMESTEP);

            for (int index=0; index < ITEMS_PER_TIMESTEP; ++index) {
                assert(data[index]["id"] == index);
                snapshots_raw.back()[index] = data[index]["statement"];
                snapshots.back()[index] = simplify(data[index]["statement"]);
            }
        }
    }

    save_snapshots();

    cout << snapshots.size() << " snapshots in total." << endl;

    if (fileExists("data/analysis/cpp-cluster/sorted_edges.json")) {
        loadSortedEdges();
    }
    else{
        for (int i=0; i<(int)snapshots.size(); ++i) {
            cout << "Processing snapshot " << i << endl;
            for (int j=0; j<ITEMS_PER_TIMESTEP; ++j) {
                for (int k=j+1; k<ITEMS_PER_TIMESTEP; ++k) {
                    testEdge(i, j, i, k);
                }
                if (i+1 < (int)snapshots.size()) {
                    for(int k=0; k<ITEMS_PER_TIMESTEP; ++k) {
                        testEdge(i, j, i+1, k);
                    }
                }
            }
        }

        saveSortedEdges();
    }

    int num_nodes = (int)snapshots.size() * ITEMS_PER_TIMESTEP;
    int max_component = 0;

    for (int thres=MAX_STR_LEN; thres >= 1 && thres >= WEIGHT_CUTOFF; --thres) {
        if (sorted_edges[thres].empty()) continue;

        int initial_num_components = DSU::num_components(num_nodes);

        for (auto p : sorted_edges[thres]) {
            int u = p.first, v = p.second;
            if (u/ITEMS_PER_TIMESTEP != v/ITEMS_PER_TIMESTEP)
                continue;
            DSU::unite(u, v);
        }

        int num_components = DSU::num_components(num_nodes);
        assert(num_components <= initial_num_components);

        map < int , int > last_snapshot_counts;
        for (int i=0; i<ITEMS_PER_TIMESTEP; ++i) {
            int label = DSU::ancestor(getNodeID((int)snapshots.size()-1, i));
            if (last_snapshot_counts.find(label) == last_snapshot_counts.end())
                last_snapshot_counts[label] = 0;
            
            last_snapshot_counts[label] += 1;
        }
        
        bool save_this = (highest_set_bit(num_components) < highest_set_bit(initial_num_components));
        for (auto p : last_snapshot_counts) {
            if (max_component < p.second) {
                if (highest_set_bit(ITEMS_PER_TIMESTEP - p.second) < highest_set_bit(ITEMS_PER_TIMESTEP - max_component))
                    save_this = 1;
                max_component = p.second;
            }
        }

        cout << thres << ": n_comp = " << num_components << " \t" << max_component << endl;
        if (save_this) {
            cout << "Saving " << thres << endl;
            saveClusters(thres);
        }
    }

    if (CROSS_SNAPSHOT_WEIGHT_CUTOFF != -1) { // Perform cross-snapshot cluster alignment
        cout << "Pre-alignment: " << DSU::num_components(num_nodes) << " clusters" << endl;

        vector<vector<int>> snapshot_labels(snapshots.size(), vector<int>(ITEMS_PER_TIMESTEP, -1));
        vector<int> cluster_sizes(num_nodes, 0);
        
        // Back up the current DSU state
        for (int i=0; i<(int)snapshots.size(); ++i) {
            for (int j=0; j<ITEMS_PER_TIMESTEP; ++j) {
                snapshot_labels[i][j] = DSU::ancestor(getNodeID(i, j));
                cluster_sizes[snapshot_labels[i][j]] += 1;
            }
        }

        // Find each cluster a predecessor in the previous snapshot, and a successor in the next snapshot
        vector<int> pred_mapped(num_nodes, 0);
        vector<int> succ_mapped(num_nodes, 0);

        for (int thres=MAX_STR_LEN; thres >= CROSS_SNAPSHOT_WEIGHT_CUTOFF; --thres) {
            if (sorted_edges[thres].empty()) continue;

            for (auto p : sorted_edges[thres]) {
                int u = p.first, v = p.second;
                if (u > v) swap(u, v);

                int u_timestep = u / ITEMS_PER_TIMESTEP, u_index = u % ITEMS_PER_TIMESTEP;
                int v_timestep = v / ITEMS_PER_TIMESTEP, v_index = v % ITEMS_PER_TIMESTEP;
                if (u_timestep == v_timestep) continue;
                assert(u_timestep + 1 == v_timestep);

                int u_label = snapshot_labels[u_timestep][u_index];
                int v_label = snapshot_labels[v_timestep][v_index];
                assert(u_label != -1 && v_label != -1);

                if (abs(cluster_sizes[u_label] - cluster_sizes[v_label]) > 8 && max(cluster_sizes[u_label], cluster_sizes[v_label]) > 2 * min(cluster_sizes[u_label], cluster_sizes[v_label])) {
                    continue;
                }

                if (!pred_mapped[v_label] && !succ_mapped[u_label]) {
                    pred_mapped[v_label] = 1;
                    succ_mapped[u_label] = 1;
                    DSU::unite(u, v);
                }
            }
        }

        cout << "Post-alignment: " << DSU::num_components(num_nodes) << " cluster" << endl;
        saveClusters(WEIGHT_CUTOFF, true, true);
    }

    return 0;
}
