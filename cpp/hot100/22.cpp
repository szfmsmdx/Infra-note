#include<vector>
using namespace std;

class Solution {
public:
    vector<string> res;
    void dfs(int left, int right, int balance, string &cur){
        if(left == 0 && right == 0 && balance == 0){
            res.push_back(cur);
            return;
        }
        if(left){
            cur += '(';
            dfs(left - 1, right, balance + 1, cur);
            cur.pop_back();
        }
        if(right && balance){
            cur += ')';
            dfs(left, right - 1, balance - 1, cur);
            cur.pop_back();
        }
    }
    vector<string> generateParenthesis(int n) {
        res.clear();
        string cur = "";
        dfs(n, n, 0, cur);
        return res;
    }
};