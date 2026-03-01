#include<string>
#include<vector>

using namespace std;

// 暴力 O n^3
// class Solution {
// public:
//     bool check(const string& s){
//         int l = 0, r = s.size() - 1;
//         while(l <= r){
//             if(s[l ++] != s[r --]) return false;
//         }
//         return true;
//     }
//     int countSubstrings(string s) {
//         int res = 0;
//         for(int i = 0; i < s.size(); ++i){
//             for(int j = i; j < s.size(); ++j){
//                 if(check(s.substr(i, j - i + 1))) res ++;
//             }
//         }        
//         return res;
//     }
// };

// dp O n^2
// class Solution {
// public:
//     int countSubstrings(string s) {
//         int n = s.size();
//         // dp[i][j] 表示 [i,j] 是否是回文的
//         vector<vector<bool>> dp(n, vector<bool>(n, false));
//         for(int i = 0; i < n; ++i) dp[i][i] = true;
//         // dp[i][j] = (dp[i+1][j-1] && s[i] == s[j]);
//         int res = n;
//         for(int i = n - 1; i >= 0; --i){
//             for(int j = i + 1; j < n; ++j){
//                 if(s[i] == s[j]){
//                     if(j - i == 1){
//                         dp[i][j] = true;
//                         res ++;
//                     } else if(dp[i+1][j-1]){
//                         dp[i][j] = true;
//                         res ++;
//                     }
//                 }
//             }
//         }
//         return res;
//     }
// };

