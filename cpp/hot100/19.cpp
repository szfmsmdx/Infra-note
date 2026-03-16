#include"ListNode.h"

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        auto cur = head;
        int l = 0;
        while(cur){
            l++;
            cur = cur->next;
        }

        ListNode* dummy = new ListNode();
        dummy->next = head;
        cur = dummy;
        for (int i = 0; i < (l - n); ++i){
            cur = cur->next;
        }
            auto p = cur->next;
        cur->next = p->next;
        p->next = nullptr;
        delete p;
        return dummy->next;
    }
};