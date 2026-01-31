#include<iostream>
#include<TreeNode.h>
#include <climits>

using namespace std;

class Solution {
private:
    int res;
public:
    int maxc(TreeNode* p){
        // 包含 p 的一条路径最大值;
        if(p->left == nullptr && p->right == nullptr){
            res = max(p->val, res);
            return p->val; 
        }

        int left_max = p->left ? maxc(p->left) : 0;
        int right_max = p->right ? maxc(p->right) : 0;
        left_max = max(left_max, 0);
        right_max = max(right_max, 0);
        res = max(res, left_max + right_max + p->val);
        return p->val + max(left_max, right_max);
    }
    int maxPathSum(TreeNode* root) {
        res = INT_MIN;
        int tmp = maxc(root);
        return res;
    }
};