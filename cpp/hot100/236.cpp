#include<TreeNode.h>
#include<iostream>
#include<unordered_map>

using namespace std;

class Solution {
public:
    unordered_map<TreeNode*, TreeNode*>h;
    unordered_map<TreeNode *, bool> vis;
    void dfs(TreeNode *root){
        if(root->left){
            h[root->left] = root;
            dfs(root->left);
        }
        if(root->right){
            h[root->right] = root;
            dfs(root->right);
        }
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 哈希表记录父节点
        if(!root) return root;
        h[root] = nullptr;
        dfs(root);
        while(h.count(p)){
            vis[p] = true;
            p = h[p];
        }
        while(h.count(q)){
            if(vis[q]) return q;
            q = h[q];
        }
        return nullptr;
    }
};