#include<iostream>
#include<vector>

using namespace std;

class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix.back().size();
        vector<vector<int>> h(m, vector<int>(n, 0)), w(m, vector<int>(n, 0));
        h[0][0] = matrix[0][0] - '0', w[0][0] = matrix[0][0] - '0';
        int res = matrix[0][0] - '0';
        for(int i = 1; i < m; ++i){
            h[i][0] = (matrix[i][0] - '0' == 0) ? 0 : h[i - 1][0] + 1;
            w[i][0] = (matrix[i][0] - '0' == 0) ? 0 : 1;
            res = max(h[i][0] * w[i][0], res);
        }
        for(int j = 1; j < n; ++j){
            h[0][j] = (matrix[0][j] - '0') == 0 ? 0 : 1;
            w[0][j] = (matrix[0][j] - '0') == 0 ? 0 : w[0][j - 1] + 1;
            res = max(res, h[0][j] * w[0][j]);
        }
        for(int i = 1; i < m; ++i){
            for(int j = 1; j < n; ++j){
                if(matrix[i][j] - '0' == 0){
                    h[i][j] = 0, w[i][j] = 0;
                }else{
                    h[i][j] = min(h[i - 1][j - 1], h[i - 1][j]) + 1;
                    w[i][j] = min(w[i - 1][j - 1], w[i][j - 1]) + 1;
                    res = max(res, h[i][j] * w[i][j]);
                }
            }
        }
        return res;
    }
};