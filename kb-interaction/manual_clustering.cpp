#include "kbutils/json.hpp"
#include <bits/stdc++.h>
using namespace std;
using json = nlohmann::json;

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

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Example usage:
    string s1, s2;
    cin >> s1 >> s2;

    auto [len1, len2, len3] = analyzeCommonSubstrings(s1, s2);

    cout << "1) Length of the single longest common substring: " << len1 << "\n";
    cout << "2) Max total length of two non-overlapping common substrings: " << len2 << "\n";
    cout << "3) Max total length of three non-overlapping common substrings: " << len3 << "\n";

    return 0;
}
