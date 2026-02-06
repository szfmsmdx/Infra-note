#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;

class Solution {
public:
    double dfs(vector<vector<double>>& g, int cur, int target, double val, vector<bool>& used){
        if(cur == target) return val;
        used[cur] = true;
        for(int next = 0; next < g[cur].size(); ++next){
            if(!used[next] && g[cur][next] != 0.0){
                double res = dfs(g, next, target, val * g[cur][next], used);
                if(res != -1.0) return res;
            }
        }
        return -1.0;
    }
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        unordered_map<string, int>s2i;
        int n = 0;
        for (auto& eq : equations) {
            if (!s2i.count(eq[0])) s2i[eq[0]] = n++;
            if (!s2i.count(eq[1])) s2i[eq[1]] = n++;
        }
        vector<vector<double>> g(n, vector<double>(n, 0.0));
        for (int i = 0; i < equations.size(); ++i) {
            int a = s2i[equations[i][0]];
            int b = s2i[equations[i][1]];
            g[a][b] = values[i];
            g[b][a] = 1.0 / values[i];
        }
        vector<double> res;
        for (auto& q : queries) {
            string a = q[0], b = q[1];
            if (!s2i.count(a) || !s2i.count(b)) {
                res.push_back(-1.0);
                continue;
            }
            int aid = s2i[a], bid = s2i[b];
            vector<bool> used(n, false); // 每次查询新建 used
            double ans = dfs(g, aid, bid, 1.0, used);
            res.push_back(ans);
        }
        return res;
    }
};