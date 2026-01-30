struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* sortList(ListNode* head) {
        return sortList(head, nullptr);
    }

    ListNode* sortList(ListNode* head, ListNode* tail){
        if(head == nullptr) return head;
        if(head->next == tail){
            // 拆成两个区间
            head->next = nullptr;
            return head;
        }
        ListNode* slow = head, *fast = head;
        while(fast != tail){
            slow = slow->next;
            fast = fast->next;
            if(fast != tail){
                fast = fast->next;
            }
        }

        ListNode* mid = slow;
        return merge(sortList(head, mid), sortList(mid, tail));
    }

    ListNode* merge(ListNode* head1, ListNode* head2){
        ListNode* prehead = new ListNode();
        ListNode* p1 = head1, * p2 = head2;
        ListNode* cur = prehead;
        while(p1 && p2){
            if(p1->val <= p2->val){
                cur->next = p1;
                p1 = p1->next;
            }else{
                cur->next = p2;
                p2 = p2->next;
            }
            cur = cur->next;
        }
        if(p1) cur->next = p1;
        if(p2) cur->next = p2;
        return prehead->next;
    }
};