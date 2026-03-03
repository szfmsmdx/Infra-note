#include<iostream>
#include<string>
#include<vector>
#include<stack>
#include<algorithm>

using namespace std;

class Solution {
public:
    string decodeString(string s) {
        stack<string> st_string;
        stack<int> st_num;
        string res = "";    // 当前字符串状态
        int num = 0;

        for(auto c : s){
            if('0' <= c && c <= '9'){
                num = num * 10 + (c - '0');
            }else if(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'){
                res += c;
            }else if(c == '['){
                st_string.push(res), res = "";
                st_num.push(num), num = 0;
            }else{
                int cnt = st_num.top(); st_num.pop();
                while(cnt--) st_string.top() += res;
                res = st_string.top();
                st_string.pop();
            }
        }

        return res;
    }
};