#include<vector>
#include<iostream>

using namespace std;

// 非定长版本滑动窗口
// 逻辑是，内层 while 循环结束后，子串 t 的每种出现次数肯定都小于等于 p 的每种出现次数
// 此时长度相等说明他们是符合条件的
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int cnt[26]{0};
        for (auto c : p) cnt[c - 'a'] ++;
        vector<int> res;
        int left = 0;
        for (int right = 0; right < s.size(); ++right){
            int c = s[right] - 'a';
            --cnt[c];
            while(cnt[c] < 0){
                cnt[s[left++] - 'a']++;
            }
            if(right - left + 1 == p.size()) res.push_back(left);
        }
        return res;
    }
};