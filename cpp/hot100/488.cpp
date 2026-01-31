#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        // 思路是把索引和数对应
        for(int i = 0; i < n; ++i){
            int a = abs(nums[i]);
            if(nums[a - 1] > 0){
                nums[a - 1] = -nums[a - 1];
            }
        }
        vector<int> res;
        for (int i = 0; i < n; ++i){
            if(nums[i] > 0)
                res.push_back(i + 1);
        }
        return res;
    }
};