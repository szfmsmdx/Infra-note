#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> res{0};
        for(int i = 1; i <= n; ++i){
            res.push_back(res[i >> 1] + (i & 1));
        }
        return res;
    }
};