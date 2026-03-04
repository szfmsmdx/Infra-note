#include"TreeNode.h"
#include<sstream>
#include<string>
using namespace std;

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if(!root) return "#";
        return to_string(root->val) + ' ' + serialize(root->left) + ' ' + serialize(root->right);
    }

    TreeNode* dfs(istringstream &s){
        string tmp;
        s >> tmp;
        if(tmp == "#") return nullptr;
        TreeNode *p = new TreeNode(stoi(tmp));
        p->left = dfs(s);
        p->right = dfs(s);
        return p;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream s(data);
        return dfs(s);
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));