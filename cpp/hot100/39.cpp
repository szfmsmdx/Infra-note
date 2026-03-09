#include<vector>
#include<algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> res;
    vector<int> cur;
    void dfs(vector<int> &candidates, int target, int cur_sum, int idx){
        if(cur_sum == target){
            res.push_back(cur);
            return ;
        }

        for(int i = idx; i < candidates.size() && candidates[i] + cur_sum <= target; ++i){
            cur.push_back(candidates[i]);
            cur_sum += candidates[i];
            dfs(candidates, target, cur_sum, i);    // 这里要传 i
            cur_sum -= candidates[i];
            cur.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        res.clear(), cur.clear();
        dfs(candidates, target, 0, 0);
        return res;
    }
};