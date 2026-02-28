#include<vector>

using namespace std;

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        // 首先肯定是数字越多越好，如果中间有 0 自己会断
        int res = INT_MIN;
        int max_v = 1, min_v = 1;
        for (int i = 0; i < nums.size(); ++i){
            if(nums[i] < 0) swap(max_v, min_v);
            max_v = max(max_v * nums[i], nums[i]); // 这里规定了如果max_v>1才会更新，所以上一个是0的话需要变成自身
            min_v = min(min_v * nums[i], nums[i]); 
            res = max(res, max_v);
        }
        return res;
    }
};