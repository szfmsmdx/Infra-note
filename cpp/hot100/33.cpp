#include<vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while(l <= r){
            int mid = (l + r) >> 1;
            if(nums[mid] == target) return mid;
            // 这里要用到 nums[r] 所以要采用闭区间搜索
            if(nums[mid] >= nums[l]){    // 左半边有序
                if(target >= nums[l] && target < nums[mid]) r = mid - 1;
                else l = mid + 1;
            }
            else if(nums[mid] <= nums[r]){
                if(target > nums[mid] && target <= nums[r]) l = mid + 1;
                else r = mid - 1;
            }
        }
        return -1;
    }
};