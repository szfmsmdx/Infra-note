#include<vector>
#include<queue>
#include<unordered_map>

using namespace std;

// 利用大根堆
// class Solution {
// public:
//     vector<int> topKFrequent(vector<int>& nums, int k) {
//         unordered_map<int, int> h;
//         priority_queue<pair<int,int>>pq;    // 小顶堆;
//         for(auto i : nums) h[i] ++;
//         for(auto p : h){
//             pq.push(pair<int, int>(-p.second, p.first));
//             if(pq.size() > k) pq.pop();
//         }
//         vector<int> res;
//         while(k--){
//             res.push_back(pq.top().second), pq.pop();
//         }
//         return res;
//     }
// };

// 桶排序
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> h;
        int max_len = 0;
        for(auto i : nums){
            h[i] ++;
            max_len = max(max_len, h[i]);
        } 
        vector<vector<int>> bucket(max_len + 1, vector<int>());
        for(auto i : h){
            bucket[i.second].push_back(i.first);
        }
        vector<int> res;
        for(int i = max_len; i && res.size() < k; --i){
            res.insert(res.end(), bucket[i].begin(), bucket[i].end());
        }
        return res;
    }
};