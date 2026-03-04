#include"TreeNode.h"
#include<string>
#include<queue>
using namespace std;

class Codec {
public:
    string res;
    void dfs(TreeNode* root){
        if(!root){
            res += "n ";
            return;
        }
        res += to_string(root->val); res +=' ';
        dfs(root->left);
        dfs(root->right);
    }

    void dfs(TreeNode* root, int idx, vector<string>& val){

    }

    string serialize(TreeNode* root) {
        res.clear();
        dfs(root);
        return res;
    }

    TreeNode* deserialize(string data) {
        vector<string> val;
        int i = 0;
        while (i < data.size()) {
            int j = i;
            while(j < data.size() && data[j] != ' ') j++;
            if(j > i){
                string s = data.substr(i, j - i);
                val.push_back(s);
            }
            i = j + 1;
        }
        if (val.empty() || val[0] == "n") return nullptr;
        TreeNode *root = new TreeNode(stoi(val[0]));
        int idx = 0;
        dfs(root, idx, val);
        return root;
    }
};