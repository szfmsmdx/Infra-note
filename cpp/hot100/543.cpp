#include<iostream>
#include<TreeNode.h>

using namespace std;

class Solution {
public:
    int res;
    int dfs(TreeNode* root){
        if(!root) return 0;
        int llen = dfs(root->left);
        int rlen = dfs(root->right);
        res = max(res, rlen + llen + 1);
        return max(llen, rlen) + 1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        res = 0;
        dfs(root);
        return res;
    }
};