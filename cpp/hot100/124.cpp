#include<iostream>
#include<TreeNode.h>
#include <climits>

using namespace std;

class Solution {
public:
    int res;
    int maxv(TreeNode* root){
        if(!root) return 0;
        int left_max = max(maxv(root->left), 0);
        int right_max = max(maxv(root->right), 0);
        res = max(res, root->val + left_max + right_max);
        return root->val + max(left_max, right_max);
    }
    int maxPathSum(TreeNode* root) {
        res = INT_MIN;
        maxv(root);
        return res;
    }
};