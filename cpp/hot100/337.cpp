#include"TreeNode.h"
#include<vector>

using namespace std;

class Solution {
public:
    vector<int> robTree(TreeNode* root){
        if(!root) return {0, 0};
        vector<int> left = robTree(root->left);
        vector<int> right = robTree(root->right);
        int norob = max(left[0], left[1]) + max(right[0], right[1]);
        int rob = root->val + left[0] + right[0];
        return {norob, rob};
    }
    int rob(TreeNode* root) {
        // 逆向思维，从下往上思考
        vector<int> res = robTree(root);
        return max(res[0], res[1]);
    }
};