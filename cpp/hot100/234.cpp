#include<ListNode.h>
#include<iostream>

using namespace std;

class Solution {
public:
    ListNode* midNode(ListNode* head) { // 得到的是中间或者中间偏右的位置
        ListNode *fast = head, *slow = head;
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
    ListNode* reverse(ListNode* head) {
        // pre 是反转后链表的头，cur指向当前链表的头
        ListNode *pre = nullptr, *cur = head;
        while(cur){
            ListNode *nxt = cur->next;
            cur->next = pre;
            pre = cur, cur = nxt;
        }
        return pre;
    }
    bool isPalindrome(ListNode* head) {
        ListNode *mid = midNode(head);
        ListNode *head2 = reverse(mid);
        while(head2){
            if(head->val != head2->val) return false;
            head = head->next, head2 = head2->next;
        }
        return true;
    }
};