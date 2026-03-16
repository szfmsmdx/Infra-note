#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> h;
        h[0] = 1;
        int res = 0, cur = 0;
        for (auto i : nums){
            cur += i;
            res += h[cur - k];
            h[cur]++;
        }
        return res;
    }
};