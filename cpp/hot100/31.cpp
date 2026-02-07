#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        // 思路是这样的，首先，初始的升序排序肯定是向上的，最终的逆序排序肯定是向下的
        // 那么某一步的过程应该是先升到一个顶端，后面是起伏的降
        //那么自然的，我们的目标是找到最后一个起伏让他更高，然后后面都是顺着往下的
        //也就是我们要从右开始找到第一个升序的相邻数对（i，j）
        //那么这个时候，(j,end)中j肯定是最大的了，那么想要更大你只能换i位置的元素
        //那么很自然的，是把(j,end)中最小的那个换给i，然后把(j,end)升序就可以了
        int n = nums.size();
        if(n <= 1) return ;
        int i = n - 2;
        while(i >= 0 && nums[i] >= nums[i + 1]) i--;
        // 这时候还是从右边遍历因为j后面肯定是降序
        if(i >= 0){
            int k = n - 1;
            while(nums[k] <= nums[i]) k --;
            swap(nums[i], nums[k]);
        }
        reverse(nums.begin() + i + 1, nums.end());
    }
};