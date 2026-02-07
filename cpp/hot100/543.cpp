#include<iostream>
#include<TreeNode.h>

using namespace std;

class Solution {
public:
    int maxl = 0;
    int dfs(TreeNode* root){   // 单边长度
        if(!root) return 0;
        int left_len = dfs(root->left);
        int right_len = dfs(root->right);
        maxl = max(maxl, left_len + right_len);
        return max(left_len, right_len) + 1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        maxl = 0;
        dfs(root);
        return maxl;
    }
};