#include"ListNode.h"
#include<vector>
#include<queue>
using namespace std;

// // 小顶堆
// class Solution {
// public:
//     struct CompareNode {
//         bool operator()(const ListNode* a, const ListNode *b) const{
//             return a->val > b->val;
//         }
//     };
//     ListNode* mergeKLists(vector<ListNode*>& lists) {
//         int n = lists.size();
//         priority_queue<ListNode*, vector<ListNode*>, CompareNode> pq;
//         for(auto head : lists){
//             if(head){
//                 pq.push(head);
//             }
//         }

//         ListNode* dummy = new ListNode();
//         auto cur = dummy;
//         while(!pq.empty()){
//             auto node = pq.top(); pq.pop();
//             if(node->next){
//                 pq.push(node->next);
//             }
//             cur->next = node;
//             cur = cur->next;
//         }

//         return dummy->next;
//     }
// };

// reduce
class Solution {
public:
    ListNode* mergeTwoList(ListNode* a, ListNode *b){
        ListNode* dummy = new ListNode();
        auto cur = dummy;
        while(a && b){
            if(a->val < b->val){
                cur->next = a;
                a = a->next;
            }else{
                cur->next = b;
                b = b->next;
            }
            cur = cur->next;
        }
        cur->next = a ? a : b;
        return dummy->next;
    }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int n = lists.size();
        if(n == 0) return nullptr;
        for (int step = 1; step < n; step *= 2){
            for (int i = 0; i < n - step; i += step * 2){
                lists[i] = mergeTwoList(lists[i], lists[i + step]);
            }
        }
        return lists[0];
    }
};