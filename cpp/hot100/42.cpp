#include<vector>
#include<stack>
using namespace std;

// class Solution {
// public:
//     int trap(vector<int>& height) {
//         int n = height.size();
//         vector<int> maxl(n, 0), maxr(n, 0);
//         maxl[0] = height[0], maxr[n - 1] = height[n - 1];
//         for(int i = 1; i < n; ++i){
//             maxl[i] = max(maxl[i - 1], height[i]);
//         }
//         for(int i = n - 2; i >= 0; --i){
//             maxr[i] = max(maxr[i + 1], height[i]);
//         }
//         int res = 0;
//         for(int i = 0; i < n; ++i){
//             res += min(maxl[i], maxr[i]) - height[i];
//         }
//         return res;
//     }
// };

class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        int l = 0, r = n - 1;
        int lmax = INT_MIN, rmax = INT_MIN;
        int res = 0;
        while(l < r){
            lmax = max(lmax, height[l]);
            rmax = max(rmax, height[r]);
            if(lmax < rmax){
                res += lmax - height[l];
                l++;
            }else{
                res += rmax - height[r];
                r--;
            }
        }
        return res;
    }
};