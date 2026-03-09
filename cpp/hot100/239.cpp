#include<vector>
#include<deque>
using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> res(n - k + 1);
        deque<int> q;

        for (int i = 0; i < n; ++i){
            while(!q.empty() && nums[q.back()] <= nums[i])
                q.pop_back();
            q.push_back(i);

            int left = i - k + 1;
            if(q.front() < left)
                q.pop_front();
            
            if(left >= 0) res[left] = nums[q.front()];
        }
        return res;
    }
};