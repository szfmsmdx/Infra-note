#include<TreeNode.h>

using namespace std;

class Solution {
public:
    TreeNode* pre;
    void flatten(TreeNode* root) {
        if(!root) return ;
        flatten(root->right);
        flatten(root->left);
        root->right = pre;
        root->left = nullptr;
        pre = root;
    }
};