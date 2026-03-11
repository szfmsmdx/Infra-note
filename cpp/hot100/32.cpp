#include<iostream>
#include<stack>
using namespace std;

// // 先来个错误思路：
// // 错误的原因是："()(()" 这个案例会输出4但实际上是2，这是对于子串所以要用栈来严格判断
// class Solution {
// public:
//     int longestValidParentheses(string s) {
//         int res = 0;
//         int left = 0, right = 0, balance = 0;
//         for (int l = 0, r = 0; r < s.size(); ++r){
//             if(s[r] == '('){
//                 left++, balance++;
//             }else{
//                 right++, balance--;
//             }
//             while(balance < 0){
//                 if(s[l] == '(') left--, balance--;
//                 else right--, balance++;
//                 l++;
//             }

//             res = max(res, r - l + 1 - balance);
//         }
//         return res;
//     }
// };

class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> st;  
        st.push(-1);    // '(' 永远合法，所以 st 头部记录的是最后一个不合法的 ')' 索引
        int res = 0;
        for (int i = 0; i < s.size(); ++i){
            if(s[i] == '(') st.push(i);
            else if(st.size() > 1){
                st.pop();
                res = max(res, i - st.top());
            }else st.top() = i;
        }
        return res;
    }
};