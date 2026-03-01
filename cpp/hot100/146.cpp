#include<unordered_map>

using namespace std;

struct DListNode{
    int key, value;
    DListNode* pre;
    DListNode* next;
    DListNode(): key(0), value(0), pre(nullptr), next(nullptr) {};
    DListNode(int key, int val): key(key), value(val), pre(nullptr), next(nullptr) {};
};

class LRUCache {
public:
    int capacity;
    unordered_map<int, DListNode*> h;
    DListNode* head = nullptr, *tail = nullptr;
    LRUCache(int capacity) {
        head = new DListNode(), tail = new DListNode();
        head->next = tail, tail->pre = head;
        this->capacity = capacity;
    }

    void move_to_head(DListNode* p){
        // 这部分要放在 get 里面写因为 put 用这段逻辑的时候 next 是空指针没有 pre 和 next
        // p->pre->next = p->next;
        // p->next->pre = p->pre;
        p->next = head->next;
        p->pre = head;
        head->next = p;
        p->next->pre = p;
    }

    void del_tail(){
        DListNode* t = tail->pre;
        t->pre->next = tail;
        tail->pre = t->pre;
        t->pre = nullptr;
        t->next = nullptr;
        h.erase(t->key);
        delete t;
    }
    
    int get(int key) {
        if(h.count(key)){
            DListNode* p = h[key];
            p->pre->next = p->next;
            p->next->pre = p->pre;
            move_to_head(p);
            return p->value;
        } return -1;
    }
    
    void put(int key, int value) {
        if(h.count(key)){
            DListNode* p = h[key];
            p->pre->next = p->next;
            p->next->pre = p->pre;
            move_to_head(p);
            p->value = value;
        } else {
            if(h.size() == capacity){
                del_tail();
            } 
            DListNode* p = new DListNode(key, value);
            move_to_head(p);
            h[key] = p;
        }
    }
};