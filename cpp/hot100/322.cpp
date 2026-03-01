#include<vector>
#include<algorithm>

using namespace std;

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // dp[i] 表示金额 i 的最少金币数
        vector<int> dp(amount + 1, INT_MAX);
        dp[0] = 0;
        for(int i = 1; i <= amount; ++i){
            for(auto c : coins){
                if(i < c) continue;
                if(dp[i - c] != INT_MAX) dp[i] = min(dp[i], dp[i - c] + 1);
            }
        }
        return dp.back() == INT_MAX ? -1 : dp.back();
    }
};