#include<iostream>
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
private:
    int capacity;
    unordered_map<int, DListNode*> h;
    DListNode* head, *tail;

public:
    LRUCache(int capacity) {
        this->capacity = capacity;
        head = new DListNode(), tail = new DListNode();
        head->next = tail;
        tail->pre = head;
    }
    
    int get(int key) {
        if(h.count(key)){
            DListNode *p = h[key];
            p->pre->next = p->next;
            p->next->pre = p->pre;
            move2head(p);
            return p->value;
        } else return -1;
    }
    
    void put(int key, int value) {
        if(h.count(key)){
            DListNode *p = h[key];
            p->pre->next = p->next;
            p->next->pre = p->pre;
            move2head(p);
            p->value = value;
        } else {
            if(h.size() < this->capacity){
                // 创建然后添加到头
                DListNode *p = new DListNode(key, value);
                move2head(p);
                h[key] = p;
            } else {
                // 删除尾
                DListNode *p = new DListNode(key, value);
                del_tail();
                move2head(p);
                h[key] = p;
            }
        }
    }

    void move2head(DListNode* p){
        p->next = head->next;
        p->pre = head;
        head->next = p;
        p->next->pre = p;
    }

    void del_tail(){
        DListNode* t = tail->pre;
        t->pre->next = t->next;
        t->next->pre = t->pre;
        t->pre = nullptr;
        t->next = nullptr;
        h.erase(t->key);
        delete t;
    }
};
