#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

class Solution {
public:
    double dfs(vector<vector<double>>& g, int ida, int idb, double cur, vector<bool>used){
        if(ida == idb) return cur;
        used[ida] = true;
        for (int i = 0; i < g[ida].size(); ++i){
            if(!used[i] && g[ida][i] != 0.0){
                double res = dfs(g, i, idb, cur * g[ida][i], used);
                if(res != -1.0)
                    return res;
            }
        }
        return -1.0;
    }
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        unordered_map<string, int> s2i;
        int cnt = 0;
        for(auto &eq : equations){
            if(!s2i.count(eq[0])) s2i[eq[0]] = cnt++;
            if(!s2i.count(eq[1])) s2i[eq[1]] = cnt++;
        }

        // 建图
        vector<vector<double>> g(cnt, vector<double>(cnt, 0.0));
        for (int i = 0; i < values.size(); ++i){
            int ida = s2i[equations[i][0]], idb = s2i[equations[i][1]];
            double v = values[i];
            g[ida][idb] = v, g[idb][ida] = 1 / v;
        }

        vector<double> res;
        for(auto q : queries){
            string a = q[0], b = q[1];
            if(!s2i.count(a) || !s2i.count(b)){
                res.push_back(-1.0);
                continue;
            }
            int ida = s2i[a], idb = s2i[b];
            vector<bool> used(cnt, false);
            double target_v = dfs(g, ida, idb, 1.0, used);
            res.push_back(target_v);
        }
        return res;
    }
};