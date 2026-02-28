#include<vector>

using namespace std;

class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int curmax = nums[0], cnt = 1;
        if(nums.size() == 1) return curmax;
        for (int i = 1; i < nums.size(); ++i) {
            if(nums[i] == curmax) cnt++;
            else{
                cnt--;
                if(cnt == 0){
                    curmax = nums[i], cnt = 1;
                }
            }
        }
        return curmax;
    }
};