#include<iostream>
#include<vector>
#include<unordered_set>

using namespace std;

class Solution {
public:
    unordered_set<string> st;
    void dfs(string &s, string &cur, int idx, int left, int right, int balance){
        // balance的作用是维护特殊情况比如 ())( 这种情况虽然 left=right=0但是不平衡
        // 维护的是当前状态是否平衡
        if(idx == s.size()){
            if(balance == 0 && left == 0 && right == 0){
                st.insert(cur);
            }
            return ;
        }

        char c = s[idx];
        if(c == '('){
            if(left > 0){   // 删除
                dfs(s, cur, idx + 1, left - 1, right, balance); 
            }
            cur.push_back(c);
            dfs(s, cur, idx + 1, left, right, balance + 1);   // 保留
            cur.pop_back();
        }else if(c == ')'){
            if(right > 0){  // 删除
                dfs(s, cur, idx + 1, left, right - 1, balance);
            }
            if(balance > 0){
                cur.push_back(c);
                dfs(s, cur, idx + 1, left, right, balance - 1);
                cur.pop_back();
            } 
        }else{
            cur.push_back(c);
            dfs(s, cur, idx + 1, left, right, balance);
            cur.pop_back();
        }
    }
    vector<string> removeInvalidParentheses(string s) {
        st.clear();
        int left = 0, right = 0;
        for(auto c:s){
            if(c == '('){
                left ++;
            }
            if(c == ')'){
                if(left > 0){
                    left --;
                }else right ++;
            }
        }
        string cur;
        dfs(s, cur, 0, left, right, 0);
        return vector<string>(st.begin(), st.end());
    }
};