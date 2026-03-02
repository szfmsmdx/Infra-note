#include<vector>
#include<numeric>
using namespace std;

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if(sum % 2) return false;
        sum = sum / 2;
        vector<int> dp(sum + 1, 0);
        for (int i = 0; i < nums.size(); ++i){
            for (int j = sum; j >= nums[i]; --j){
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return dp.back() == sum;
    }
};