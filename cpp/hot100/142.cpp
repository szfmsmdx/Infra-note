#include"ListNode.h"

using namespace std;

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        // 可能在环内相遇
        if(!head || !head->next) return nullptr;
        ListNode *slow = head, *fast = head;
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) {  // 有环
                ListNode *p = head;
                while(p != fast) {
                    p = p->next;
                    fast = fast->next;
                }
                return p;
            }
        }
        return nullptr;
    }
};