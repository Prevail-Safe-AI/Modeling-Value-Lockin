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
vector<vector<string>> snapshots;

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

void testEdge(int timestep_x, int index_x, int timestep_y, int index_y) {
    int weight = getWeight(snapshots[timestep_x][index_x], snapshots[timestep_y][index_y]);
    if (weight > MAX_STR_LEN) {
        cout << "Anomaly: " << weight << endl;
        weight = MAX_STR_LEN;
    }
    sorted_edges[weight].push_back(make_pair(getNodeID(timestep_x, index_x), getNodeID(timestep_y, index_y)));
}

void saveClusters(int stamp) {
    json data = json::array();
    for (int i=0; i<(int)snapshots.size(); ++i) {
        json current_snapshot = json::array();
        for (int j=0; j<ITEMS_PER_TIMESTEP; ++j) {
            json current_item = {
                {"id", j},
                {"statement", snapshots[i][j]},
                {"label", DSU::ancestor(getNodeID(i, j))}
            };
            current_snapshot.push_back(current_item);
        }
        json current_object = {
            {"snapshot_id", i},
            {"content", current_snapshot}
        };
        data.push_back(current_object);
    }
    string path = "data/analysis/cpp-cluster/" + to_string(stamp) + ".json";
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
}

int highest_set_bit(int v){
    int c = 0;
    while(v) {
        v >>= 1;
        c += 1;
    }
    return c;
}

int main(){
    feature_test();
    DSU::init();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector < string > directories = {
        "data/runs/ICL-run3-20250125-151456/round000"
    };

    for (auto directory : directories) {
        for (int turn=1; ; ++turn) {
            string turn_num = to_string(turn);
            if (turn < 10) {
                turn_num = "0" + turn_num;
            }
            
            string path = directory + "/knowledge-turn" + turn_num + ".json";
            if (!fileExists(path)) break;

            cout << "Reading " << path << endl;
            ifstream f(path);
            json data = json::parse(f);

            snapshots.emplace_back(ITEMS_PER_TIMESTEP);

            for (int index=0; index < ITEMS_PER_TIMESTEP; ++index) {
                assert(data[index]["id"] == index);
                snapshots.back()[index] = data[index]["statement"];
            }
        }
    }

    cout << snapshots.size() << " snapshots in total." << endl;

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

    int num_nodes = (int)snapshots.size() * ITEMS_PER_TIMESTEP;
    int max_component = 0;

    for (int thres=MAX_STR_LEN; thres >= 1; --thres) {
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

    return 0;
}
