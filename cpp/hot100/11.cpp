#include<vector>
using namespace std;

// 暴力维护单一节点：O(n^2)
// 优化到双指针
// 那么双指针怎么移动呢？答案是移动小的那个，因为移动大的那个，小的不会改变，且距离减少，那么答案大概率减少

class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int res = 0;
        while(l < r){
            res = max(res, min(height[l], height[r]) * (r - l));
            if(height[l] > height[r]) r++;
            else l--;
        }
        return res;
    }
};