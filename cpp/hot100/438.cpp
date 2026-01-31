#include<vector>
#include<iostream>

using namespace std;

class Solution {
public:
    bool check(vector<int>a){
        for(auto i:a){
            if(i)
                return false;
        }
        return true;
    }
    vector<int> findAnagrams(string s, string p) {
        vector<int> h(26, 0);
        for (auto c:p)
            h[c - 'a']++;

        int left = 0, right = p.size() - 1;
        vector<int> res;
        if(p.size() > s.size()) return res;
        for (int i = 0; i <= right; ++i){
            h[s[i] - 'a']--;
        }

        while(right < s.size()){
            if(check(h)){
                res.push_back(left);
            }

            h[s[left] - 'a']++, left++;
            ++right;
            if(right < s.size()) h[s[right] - 'a']--;
        }

        return res;
    }
};
