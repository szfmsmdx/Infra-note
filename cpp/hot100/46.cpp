#include<vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> res;
    void dfs(vector<int>& nums, vector<int> &cur, vector<bool>& used){
        if(cur.size() == nums.size()){
            res.push_back(cur);
            return;
        }

        for (int i = 0; i < nums.size(); ++i){
            if(!used[i]){
                used[i] = true;
                cur.push_back(nums[i]);
                dfs(nums, cur, used);
                cur.pop_back();
                used[i] = false;
            }
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        res.clear();
        vector<int> cur;
        vector<bool> used(nums.size(), false);
        dfs(nums, cur, used);
        return res;
    }
};