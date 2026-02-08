#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        // 维护区间 [left, right]
        // 其中，从左往右遍历最右的right，使得right比当前最大值小
        // 从右往左遍历最左的left使得left比当前最小值大
        int left = -1, right = -1;
        int cur_max = nums[0], cur_min = nums[nums.size() - 1];
        for(int i = 1; i < nums.size(); ++i){
            if(nums[i] < cur_max) right = i;
            else cur_max = nums[i];
        }
        for(int j = nums.size() - 2; j >= 0; --j){
            if(nums[j] > cur_min) left = j;
            else cur_min = nums[j];
        }
        return right == -1 ? 0 : right - left + 1;
    }
};