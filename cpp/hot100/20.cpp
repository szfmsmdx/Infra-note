#include<iostream>
#include<stack>
#include<unordered_map>
using namespace std;

class Solution {
public:
    unordered_map<char, char> h { { ')', '(' }, { ']', '[' }, { '}', '{' } };
    bool isValid(string s) {
        stack<char> st;
        for (auto c : s) {
            if(c == '(' || c == '[' || c == '{') st.push(c);
            else{
                if(st.empty()) return false;
                else if(st.top() != h[c]) return false;
                else st.pop();
            }
        }
        return st.empty();
    }
};