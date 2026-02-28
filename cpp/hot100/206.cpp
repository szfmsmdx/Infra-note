#include"ListNode.h"
#include<iostream>

using namespace std;

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = new ListNode();
        while(head){
            ListNode *p = head;
            head = head->next;
            p->next = pre->next;
            pre->next = p;
        }
        return pre->next;
    }
};