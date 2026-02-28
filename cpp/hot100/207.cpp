#include<iostream>
#include<vector>
#include<queue>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> in(numCourses, 0);
        vector<vector<int>> out(numCourses, vector<int>());
        for(auto i : prerequisites){
            out[i[1]].push_back(i[0]);
            in[i[0]]++;
        }

        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (in[i] == 0) q.push(i);
        }

        while(!q.empty()){
            int cur = q.front(); q.pop();
            for (auto i : out[cur]) {
                in[i]--;
                if(in[i] == 0) q.push(i);
            }
        }

        for(int i = 0; i < numCourses; ++i){
            if(in[i]) return false;
        }

        return true;
    }
};