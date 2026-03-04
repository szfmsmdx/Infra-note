#include<vector>

using namespace std;

// class Solution {
// public:
//     int maxProfit(vector<int>& prices) {
//         vector<vector<int>> dp(prices.size(), vector<int>(3, 0));
//         // have，nohave，冷冻;
//         dp[0][0] = -prices[0];
//         // 这里只依赖昨天的状态，所以是可以优化空间复杂度的
//         for (int i = 1; i < prices.size(); ++i){
//             dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
//             dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
//             dp[i][2] = max(dp[i - 1][1], dp[i - 1][2]);
//         }
//         return max(dp.back()[1], dp.back()[2]);
//     }
// };


class Solution {
public:
    int maxProfit(vector<int>& prices) {
        vector<vector<int>> dp(prices.size(), vector<int>(3, 0));
        // have，nohave，冷冻;
        int have = -prices[0];
        int nohave = 0, cold = 0;
        for (int i = 1; i < prices.size(); ++i){
            int pre_have = have, pre_nohave = nohave, pre_cold = cold;
            have = max(pre_have, pre_cold - prices[i]);
            nohave = max(pre_nohave, pre_have + prices[i]);
            cold = max(pre_nohave, pre_cold);
        }
        return max(nohave, cold);
    }
};