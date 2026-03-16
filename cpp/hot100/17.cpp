#include<vector>
#include<iostream>
#include<unordered_map>
using namespace std;

class Solution {
public:
    vector<string> res;
    string cur;
    unordered_map<char, vector<char>> h {
        { '2', { 'a', 'b', 'c' } },
        { '3', { 'd', 'e', 'f' } },
        { '4', { 'g', 'h', 'i' } },
        { '5', { 'j', 'k', 'l' } },
        { '6', { 'm', 'n', 'o' } },
        { '7', { 'p', 'q', 'r', 's' } },
        { '8', { 't', 'u', 'v' } },
        { '9', { 'w', 'x', 'y', 'z' } }
    };
    void dfs(string digits, int idx){
        if(idx == digits.size()){
            res.push_back(cur);
            return;
        }
        for(auto c : h[digits[idx]]){
            cur += c;
            dfs(digits, idx + 1);
            cur.pop_back();
        }
        return;
    }
    vector<string> letterCombinations(string digits) {
        res.clear();
        dfs(digits, 0);
        return res;
    }
};