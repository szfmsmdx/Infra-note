#include<iostream>
#include<unordered_map>

using namespace std;

class Solution {
public:
    // unordered_map<char, int> h;
    // bool valid(unordered_map<char, int>& tcnt){
    //     for(auto p:tcnt){
    //         if(h[p.first] < p.second) return false;
    //     }
    //     return true;
    // }
    // string minWindow(string s, string t) {
    //     int l = 0, r = 0;
    //     if(t.size() > s.size()) return "";
    //     string res = "";
    //     int minl = INT_MAX;
    //     unordered_map<char, int>tcnt;
    //     for(auto c:t) tcnt[c]++;
    //     for(;r < s.size(); ++r){
    //         h[s[r]]++;
    //         if(!valid(tcnt)) continue;
    //         while(valid(tcnt) && l <= r) h[s[l++]]--;
    //         l = l - 1, h[s[l]] ++;
    //         if(r - l + 1 < minl){
    //             res = s.substr(l, r - l + 1);
    //             minl = r - l + 1;
    //         }
    //     }
    //     return res;
    // }
    string minWindow(string s, string t) {
        if (t.empty()) return "";
        if (s.size() < t.size()) return "";

        // 优化写法，无需反复 valid 维护当前是否 valid 的状态即可
        unordered_map<char, int> need;
        for(char c:t) need[c]++;
        unordered_map<char, int> window;
        int have = 0;   // 维护当前的种类
        int need_cnt = need.size(); // 需要的种类

        string res = "";
        int minl = INT_MAX, min_start = 0;
        int l = 0;
        for(int r = 0; r < s.size(); ++r){
            char c = s[r];
            window[c] ++;

            if(need[c] && window[c] == need[c]){
                have ++;
            }

            // 收缩
            while(have == need_cnt && l <= r){
                int cur_len = r - l + 1;
                if(cur_len < minl){
                    minl = cur_len;
                    min_start = l;
                }

                char l_char = s[l];
                window[l_char]--;

                if(need.count(l_char) && window[l_char] < need[l_char]){
                    have--;
                }
                l++;
            }
        }

        return (minl == INT_MAX) ? "" : s.substr(min_start, minl);
    }
};

int main(){
    Solution solution;
    string s = "a", t = "a";
    string res = solution.minWindow(s, t);
    cout << res << endl;
}