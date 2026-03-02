#include<vector>

using namespace std;

class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        for (int i = 0; i < nums.size(); ++i){
            int a = abs(nums[i]);
            if (nums[a - 1] > 0) nums[a - 1] = -nums[a - 1];
        }
        vector<int> res;
        for (int i = 0; i < nums.size(); ++i){
            if(nums[i] > 0) res.push_back(i + 1);
        }
        return res;
    }
};