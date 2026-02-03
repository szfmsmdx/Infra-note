#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

// 思路是：
// 用哈希表实现 O（1）查找
// 用左端点来判断序列长度
// 这个每个数字最多只会被访问两次，是O（n）的最优解

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int, int>h;
        int res = 0;
        for(auto i:nums) h[i] = 1;
        for(auto i:h){
            if(h.count(i.first - 1)){   // 说明不是左端点
                continue;
            }else{
                int cur = i.first, cur_len = 1;
                while(h.count(cur + 1)){
                    cur ++, cur_len ++;
                }
                res = max(res, cur_len);
            }
        }
        return res;
    }
};