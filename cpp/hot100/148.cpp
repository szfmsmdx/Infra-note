#include"ListNode.h"
using namespace std;

class Solution {
public:
    ListNode* get_mid(ListNode* head){
        ListNode* slow = head, *fast = head, *pre = head;
        while(fast && fast->next){
            pre = slow;
            slow = slow->next, fast = fast->next->next;
        }
        pre->next = nullptr;
        return slow;
    }
    ListNode* merge(ListNode* head, ListNode* head2){
        ListNode* res = new ListNode();
        ListNode* cur = res;
        while(head && head2){
            if(head->val < head2->val){
                cur->next = head;
                head = head->next;
            } else {
                cur->next = head2;
                head2 = head2->next;
            }
            cur = cur->next;
        }
        if(head) cur->next = head;
        if(head2) cur->next = head2;
        return res->next;
    }
    ListNode* sortList(ListNode* head) {
        if(!head || !head->next) return head;
        ListNode* head2 = get_mid(head);
        head = sortList(head);
        head2 = sortList(head2);
        return merge(head, head2);
    }
};