# 可能不过根节点问题
如：[lc543](https://leetcode.cn/problems/diameter-of-binary-tree/?envType=problem-list-v2&envId=2cktkvj)

这种题目的问题是，最优解不一定过根节点，可能藏在子数组中，比如这个例子：

```text
      1
     / \
    2   3
   / \
  4   5
 /     \
6       7
         \
          8
```

那么这个时候，最长的路径当然是以 2 为根的这个子树

对于这种问题可以有一个较为通用的解题模板，主要思路是采取树的后序遍历，这样你在遍历的时候提前获取了两棵子树的一些信息，再汇总到根节点上，就会覆盖掉所有的情况，模板如下：
```cpp
class Solution {
public:
    int global_ans = /* 初始值，如 0, INT_MIN 等 */;

    // 返回：从当前节点向下延伸的“最佳单边路径”信息（不能拐弯！）
    ReturnType dfs(TreeNode* node) {
        if (!node) return /* base value */;
        
        auto left = dfs(node->left);   // 左子树返回的单边信息
        auto right = dfs(node->right); // 右子树返回的单边信息

        // 以 node 为“顶点”，拼接左右，形成完整路径
        int fullPath = combine(left, right, node->val);
        global_ans = max(global_ans, fullPath);

        // 只能选一边（或都不选）+ 当前节点，构成向上传递的单边路径
        return extend(max(left, right), node->val);
    }

    int solve(TreeNode* root) {
        global_ans = /* reset */;
        dfs(root);
        return global_ans;
    }
};
```

对应题型有：
- 路径长度（边数 or 节点数）
    - 代表题：lc543
- 路径和最大（可含负数）
    - 代表题：lc124
- 路径上所有节点值相同
    - 代表题：lc687
- 路径满足某种性质（如和为 target）
    - 代表题 lc437
    - 这种题目通常用前缀和 + 哈希表来处理，是自顶向下 + 回溯

# 几种遍历方式的选择
- 前序遍历（处理当前节点不依赖于子树）
    - 复制/序列化一棵树
    - 构建表达式树（前缀表达式）
    - 自顶向下的信息传递
    - 寻找满足条件的路径
- 中序遍历（适合有序的情况，左边处理完再处理当前节点）
    - BST 相关问题
    - 表达树中缀表达式
- 后序遍历
    - 自底向上信息传递
    - 子树的比较、合并、剪枝
- 层序遍历
    - 求树宽、层级遍历、逐层打印
    - 最短路径
    - 序列化、反序列化