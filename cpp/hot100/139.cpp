#include<vector>
#include<string>

using namespace std;

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        // dp[i] 表示长度为 i 的子串能否拼成
        vector<bool> dp(s.size() + 1, false); dp[0] = true;
        for(int i = 0; i < s.size(); ++i){
            for(auto word : wordDict){
                if(i + 1 >= word.size() && dp[i + 1 - word.size()] && !dp[i + 1]){
                    // 这里是因为 i - word.size() 是上一个结尾，那么结尾已经被判断过了，所以要取下一个
                    if(word == s.substr(i - word.size() + 1, word.size())){
                        dp[i + 1] = true;
                    }
                }
            }
        }
        return dp.back();
    }
};