#include<vector>
#include<unordered_map>
using namespace std;

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> h;
        for(auto &s:strs){
            string tmp = s;
            sort(tmp.begin(), tmp.end());
            h[tmp].push_back(s);
        }
        vector<vector<string>> res(h.size());
        int cnt = 0;
        for(auto i : h){
            res[cnt++] = i.second;
        }
        return res;
    }
};