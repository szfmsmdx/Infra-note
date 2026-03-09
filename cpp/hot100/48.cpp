#include<vector>
using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        // 只需要转动 1/4 矩阵就行
        int n = matrix.size();
        for (int i = 0; i < n / 2; ++i){
            for (int j = 0; j < (n + 1) / 2; ++j){
                // 为了方便算一下对应的坐标
                int i2 = j, j2 = n - i - 1;
                int i3 = n - j - 1, j3 = i;
                int i4 = n - i - 1, j4 = n - j - 1;
                swap(matrix[i3][j3], matrix[i][j]);
                swap(matrix[i3][j3], matrix[i4][j4]);
                swap(matrix[i4][j4], matrix[i2][j2]);
            }
        }
    }
};