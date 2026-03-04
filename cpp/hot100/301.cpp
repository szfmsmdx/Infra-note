#include<iostream>
#include<vector>
#include<unordered_set>

using namespace std;

// class Solution {
// public:
//     vector<string> res;
//     void dfs(string &s, string cur, int idx, int cnt){
//         if(idx == s.size()){
//             if(cnt == 0) res.push_back(cur);
//             return;
//         }

//         if(s[idx] == '(' && cnt > 0) {
//             dfs(s, cur, idx + 1, cnt - 1);      // 删
//             dfs(s, cur + s[idx], idx + 1, cnt); // 不删
//         }else if(s[idx] == ')' && cnt < 0) {
//             dfs(s, cur, idx + 1, cnt + 1);
//             dfs(s, cur + s[idx], idx + 1, cnt);
//         }else dfs(s, cur + s[idx], idx + 1, cnt);

//         return;
//     }
//     vector<string> removeInvalidParentheses(string s) {
//         int cnt = 0;
//         for(auto c : s){
//             if(c == '(') cnt ++;
//             else if(c == ')') cnt--;
//         }
//         dfs(s, "", 0, cnt);
//         return res;
//     }
// };
// 贴一个错版，两个问题：
// 1. 没有去重 —— 好解决，用 unordered_set 即可解决
// 2. cnt 来计数的问题是可能出现这种情况 )( 这是不能作为一个平衡的配对的，用 balance 维护当前路径的左括号数量
// 3. 还是遇到 )(a 这种情况，应该把 )、( 都给删了，所以需要单独记录左右括号的删除个数

class Solution {
public:
    unordered_set<string> res;
    // void dfs(string &s, string cur, int idx, int left, int right, int balance){
    // 上面这么写简单，直接传入 cur + c 就可以，但是开销比较大，取引用开销更小
    void dfs(string &s, string &cur, int idx, int left, int right, int balance){
        if(idx == s.size()){
            if(balance == 0 && left == 0 && right == 0) res.insert(cur);
            return;
        }

        char c = s[idx];
        if(c == '('){
            if(left > 0) dfs(s, cur, idx + 1, left - 1, right, balance);
            cur.push_back(c);
            dfs(s, cur, idx + 1, left, right, balance + 1);
            cur.pop_back();
        }else if(c == ')'){
            if(right > 0) dfs(s, cur, idx + 1, left, right - 1, balance);
            if(balance){
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
    vector<string> removeInvalidParentheses(string s)
    {
        int left = 0, right = 0;
        for(auto i : s){
            if(i == '(') left++;
            else if(i == ')'){
                left ? left-- : right++;
            }
        }
        string cur = "";
        dfs(s, cur, 0, left, right, 0);
        return vector<string>(res.begin(), res.end());
    }
};