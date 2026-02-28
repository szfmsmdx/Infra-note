#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    void dfs(vector<vector<char>>& grid, int i, int j){
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid.back().size()) return ;
        if (grid[i][j] != '1') return ;

        grid[i][j] = '2';

        dfs(grid, i, j - 1);
        dfs(grid, i, j + 1);
        dfs(grid, i - 1, j);
        dfs(grid, i + 1, j);

        return;
    }
    int numIslands(vector<vector<char>>& grid) {
        if(grid.size() == 0 || grid.back().size() == 0) return 0;
        int res = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid.back().size(); ++j) {
                if(grid[i][j] == '1'){
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }
};