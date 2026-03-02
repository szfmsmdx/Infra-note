#include<vector>
#include<numeric>
using namespace std;

// 首先我们能够拿到 sum nums全部和
// 那么也就是说，被选为正数、负数的和分别记作 a、b
// 那么 a + b = t, a - b = s
// 自然可以求得 a = (s + t) / 2, b = (t - s) / 2
// 也就是说我们要选出来一些元素让他等于 (t - s) / 2 或者 (t + s) / 2
// 这也就是 0-1 背包问题了
// 注意一个点是需要反向遍历 dp
// 因为是 0-1 背包，你选择物品的时候，如果正向遍历的话
// 对一个价值为 n 的物品来说，dp[n] 会取到一次物品, dp[n + n] 是第二次取物品了,就变成了完全背包问题


class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if((sum - target) % 2) return 0;    // 奇数不可能取到;
        int plus = (sum + target) / 2;
        if(plus < 0) return 0;
        vector<int> dp(plus + 1, 0); dp[0] = 1;
        for (int i = 0; i < nums.size(); ++i){
            for (int j = plus; j >= nums[i]; --j){
                dp[j] += dp[j - nums[i]];
            }
        }

        return dp.back();
    }
};