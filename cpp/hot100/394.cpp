#include<iostream>
#include<string>
#include<vector>
#include<stack>
#include<algorithm>

using namespace std;

class Solution {
public:
    string decodeString(string s) {
        stack<string> st1;
        stack<int> st2;
        string res = "";
        int num = 0;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] >= '0' && s[i] <= '9'){ // 是数字
                num = num * 10 + (s[i] - '0');
            }else if(s[i] >= 'a' && s[i] <= 'z' || s[i] >= 'A' && s[i] <= 'Z'){
                res = res + s[i];
            }else if(s[i] == '['){
                st1.push(res), res = "";
                st2.push(num), num = 0;
            }else{
                int times = st2.top();
                st2.pop();
                while(times--){
                    st1.top() += res;
                }
                res = st1.top();
                st1.pop();
            }
        }

        return res;
    }
};