#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int maxCoins(vector<int>& nums) {
        // 思路是这样，首先 n 在 300 的范围，应该是 O(n^3) 的做法
        // 其次，我们考虑区间 (i, j) 这里的意思是，不戳破两端 i j 这两个气球
        // 那么 ij 这个开区间的最大值应该是多少呢？DP的状态转移方程应该是
        // max(dp[i,k] * nums[k] * dp[k,j])
        // 其中我们把第 k 个气球当做是最后一个戳破的气球
        int n = nums.size();
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 1));
        for (int i = 1; i <= n; ++i){
            for (int j = i; j <= n; ++j){
                if(i == j)
                    dp[i][j] = 1;
                else{
                    
                }
            }
        }
        return dp[1][n];
    }
};