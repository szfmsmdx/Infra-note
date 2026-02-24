#include<iostream>
#include<vector>
#include<stack>

using namespace std;

class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        // 求最大，维护单减栈
        vector<int> res(temperatures.size(), 0);
        stack<int> st;
        st.push(0);
        for (int i = 1; i < temperatures.size(); ++i){
            while(!st.empty() && temperatures[i] > temperatures[st.top()]){
                int idx = st.top();
                res[idx] = i - idx;
                st.pop();
            }
            st.push(i);
        }
        return res;
    }
};