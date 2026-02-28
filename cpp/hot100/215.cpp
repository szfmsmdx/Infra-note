#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int quick_sort(vector<int>& nums, int l, int r, int k){
        if(l == r) return nums[l];
        int i = l - 1, j = r + 1, x = nums[i + j >> 1];
        while(i < j){
            do i++; while(nums[i] > x);
            do j--; while(nums[j] < x);
            if(i < j) swap(nums[i], nums[j]);
        }
        int ll = j - l + 1;
        if(ll >= k) return quick_sort(nums, l, j, k);
        return quick_sort(nums, j + 1, r, k - ll);
    }
    int findKthLargest(vector<int>& nums, int k) {
        // 归并排序每次扔掉一半，复杂度从 nlogn 将至 n
        return quick_sort(nums, 0, nums.size() - 1, k);
    }
};