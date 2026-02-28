#include<iostream>

using namespace std;

class Node {
public:
    Node *next[26];
    bool isEnd;
    Node(){
        isEnd = false;
        for (int i = 0; i < 26; ++i) next[i] = nullptr;
    }
    ~Node(){
        for (int i = 0; i < 26; ++i){
            if(next[i]){
                delete next[i];
                next[i] = nullptr;
            }
        }
    }
};

class Trie
{
public:
    Node *root = nullptr;
    Trie() {
        root = new Node();
    }

    void insert(string word) {
        Node *cur = root;
        for (int i = 0; i < word.size(); ++i){
            int idx = word[i] - 'a';
            if(!cur->next[idx]) cur->next[idx] = new Node();
            if(i == word.size() - 1) cur->next[idx]->isEnd = true;
            cur = cur->next[idx];
        }
    }

    bool search(string word) {
        Node *cur = root;
        for (int i = 0; i < word.size(); ++i){
            int idx = word[i] - 'a';
            if(!cur->next[idx]) return false;
            if(i == word.size() - 1 && cur->next[idx]->isEnd) return true;
            cur = cur->next[idx];
        }
        return false;
    }

    bool startsWith(string prefix) {
        Node *cur = root;
        for (int i = 0; i < prefix.size(); ++i){
            int idx = prefix[i] - 'a';
            if(!cur->next[idx]) return false;
            cur = cur->next[idx];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */