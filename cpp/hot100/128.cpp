#include<vector>
#include<unordered_map>

using namespace std;

// 首先想的是，维护当前最大值也就是 nums[i]:对应的长度
// 但是如果左端点添加了新元素那么长度是不会更新的
// 所以我们需要维护左端点和最大值
// 那么如果你是左端点我怎么O1一直往后找呢？
// 所以要先预存所有端点的信息

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int, int> h;
        for(auto i:nums) h[i] = 1;
        int res = 0;
        for(auto i:h){
            if(h.count(i.first - 1)){
                continue;
            }else{  // 是左端点
                int next = i.first + 1;
                while(h.count(next)){
                    i.second ++;
                    next ++;
                }
                res = max(res, i.second);
            }
        }
        return res;
    }
};