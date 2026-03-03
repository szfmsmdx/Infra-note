#include<iostream>
#include<vector>

using namespace std;

// 思路是这样，首先 n 在 300 的范围，应该是 O(n^3) 的做法
// 其次，我们考虑区间 (i, j) 这里的意思是，不戳破两端 i j 这两个气球
// 那么 ij 这个开区间的最大值应该是多少呢？DP的状态转移方程应该是
// max(dp[i,k] * nums[k] * dp[k,j])
// 其中我们把第 k 个气球当做是最后一个戳破的气球

class Solution {
public:
    int maxCoins(vector<int>& nums) {
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for(int l = 3; l <= n; ++l){
            for(int i = 0; i + l <= n; ++i){
                int j = i + l - 1;
                for(int k = i + 1; k < j; ++k){
                    dp[i][j] = max(
                        dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[j] * nums[k]
                    );
                }
            }
        }
        return dp[0][n - 1];
    }
};