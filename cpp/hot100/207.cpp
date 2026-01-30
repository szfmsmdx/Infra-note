#include<iostream>
#include<vector>
#include<queue>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> innode(numCourses, 0);      // 记录入度;
        vector<vector<int>> g(numCourses, vector<int>{});  // 记录边
        for(int i = 0; i < prerequisites.size(); ++i){
            int a = prerequisites[i][0];
            int b = prerequisites[i][1];
            innode[a] ++;
            g[b].push_back(a);
        }

        queue<int> q;
        for(int i = 0; i < numCourses; ++i){
            if(innode[i] == 0){
                q.push(i);
            }
        }

        int finish = 0;             // 能退出队列的节点
        while(!q.empty()){
            ++finish;
            int cur = q.front(); q.pop();
            for(auto i:g[cur]){
                --innode[i];
                if(innode[i] == 0){
                    q.push(i);
                }
            }
        }

        return finish == numCourses;
    }
};