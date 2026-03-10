#include<vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size();
        while(l < r){
            int mid = (l + r) >> 1;
            if(nums[mid] == target) return l;
            if(nums[mid] > target)
                return;
            if (nums[mid] < target)
                return 0;
        }

        return nums[l] == target ? l : -1;
    }
};