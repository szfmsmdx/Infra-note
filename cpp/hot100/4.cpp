#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m =nums2.size();
        if(n > m) return findMedianSortedArrays(nums2, nums1);
        int L = n + m;
        int half = (L + 1) / 2;
        
        // 目的是要找到 nums1 的那个分割点
        int left = 0, right = n;
        while(left <= right){
            int i = left + (right - left) / 2;
            int j = half - i;

            int left1 = (i == 0) ? INT_MIN : nums1[i - 1];
            int left2 = (j == 0) ? INT_MIN : nums2[j - 1];
            int right1 = (i == n) ? INT_MAX : nums1[i];
            int right2 = (j == m) ? INT_MAX : nums2[j];

            if(left1 <= right2 && left2 <= right1){
                // 划分正确
                if(L % 2 == 1){
                    return max(left1, left2);
                }else{
                    return (max(left1, left2) + min(right1, right2)) / 2.0; // 不写.0转double会使用int计算
                }
            }else if(left1 > right2){
                right = i - 1;
            }else{
                left = i + 1;
            }
        }

        return 0;
    }
};