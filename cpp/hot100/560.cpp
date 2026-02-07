#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) { // 看到 target 要想到哈希表，子数组就考虑前缀和
        unordered_map<int, int> h{{0, 1}};
        int res = 0, s = 0;
        for(auto x:nums){
            s += x;
            res += h.count(s - k) ? h[s - k] : 0;
            h[s] ++;
        }
        return res;
    }
};