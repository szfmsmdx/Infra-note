#include<vector>
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int have = -prices[0];
        int nohave = 0;
        for(int i = 1; i < prices.size(); ++i){
            have = max(have, -prices[i]);
            nohave = max(have + prices[i], nohave);
        }   
        return nohave;     
    }
};