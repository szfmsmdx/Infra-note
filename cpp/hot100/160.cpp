#include<ListNode.h>
#include<iostream>

using namespace std;

// 主要的思路就是找到两个链表长度的差值即可
// 那么可以使用双指针以一个比较优雅的方式来实现
// 实际上就是在尾端补上了另一条链的非公共部分
// 假设两条链的公共部分是c，那么两条链分别是：a+c、b+c
// a+c -> 补上 b
// b+c -> 补上 a

class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB)
            return nullptr;
        ListNode *pa = headA, *pb = headB;
        while(pa != pb){
            pa = (pa == nullptr) ? headB : pa->next;
            pb = (pb == nullptr) ? headA : pb->next;
        }
        return pa;
    }
};