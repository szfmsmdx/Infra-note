#include<TreeNode.h>
#include<unordered_map>
#include<iostream>

using namespace std;

// 朴素思路：
// dfs
// class Solution {
// public:
//     int dfs(TreeNode* p, long long targetSum){
//         // 必须包含 p 节点
//         if(!p) return 0;
//         int res = 0;
//         if(p->val == targetSum) res = 1;
//         res += dfs(p->left, targetSum - p->val);
//         res += dfs(p->right, targetSum - p->val);
//         return res;
//     }
//     int pathSum(TreeNode* root, int targetSum) {
//         if(!root) return 0;
//         return dfs(root, targetSum) + pathSum(root->left, targetSum) + pathSum(root->right, targetSum);
//     }
// };

// 前缀和：我们可以记录 root 到 p 节点的路径的前缀和
// 那么问题就转化成了两数之和：给定一个 targetsum 以及 p-1 到 p 的值，直接查哈希表就可以了

class Solution {
public:
    unordered_map<long long, int> prefix;
    int pathSum(TreeNode* root, int targetSum) {
        
    }
};