#include<vector>
using namespace std;

// class Solution {
// public:
//     bool bi_search(vector<int> &cur, int target, int left, int right){
//         while(left < right){
//             int mid = (left + right) >> 1;
//             if(cur[mid] == target) return true;
//             else if(cur[mid] > target) right = mid;
//             else left = mid + 1;
//         }
//         return false;
//     }
//     bool searchMatrix(vector<vector<int>>& matrix, int target) {
//         int m = matrix.size(), n = matrix.back().size();
//         for (int i = 0; i < m; ++i){
//             if(matrix[i][0] > target) return false;
//             vector<int> &cur = matrix[i];
//             if(bi_search(cur, target, 0, n)) return true;
//         }
//         return false;
//     }
// };

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix.back().size();
        int x = 0, y = n - 1;
        while(x < m && y >= 0){
            if(matrix[x][y] == target) return true;
            else if(matrix[x][y] > target) y--;
            else x ++;
        }
        return false;
    }
};