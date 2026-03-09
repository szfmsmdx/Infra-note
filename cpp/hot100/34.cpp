#include<vector>
using namespace std;

class Solution {
public:
    int left_search(vector<int>& nums, int l, int r, int target){
        while(l < r){
            int mid = (l + r) >> 1;
            if(nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        return l;
    }

    int right_search(vector<int>& nums, int l, int r, int target){
        while(l < r){
            int mid = (l + r) >> 1;
            if(nums[mid] > target) r = mid;
            else l = mid + 1;
        }
        return l - 1;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.size() == 0) return {-1, -1};
        int l = left_search(nums, 0, nums.size(), target);
        int r = right_search(nums, 0, nums.size(), target);
        if(l >= nums.size() || nums[l] != target) return {-1, -1};
        return {l, r};
    }
};