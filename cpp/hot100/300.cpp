#include<vector>
using namespace std;

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);
        int res = INT_MIN;
        for (int j = 1; j < n; ++j){
            for (int i = 0; i < j; ++i){
                if(nums[j] > nums[i]) {
                    dp[j] = max(dp[j], dp[i] + 1);
                    res = max(res, dp[j]);
                }
            }
        }
        return res;
    }
};