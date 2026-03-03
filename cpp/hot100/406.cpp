#include<vector>
#include<list>

using namespace std;

// 题意没太明白...
// 大体思路应该是根据身高降序，相同的话按照 k 升序排列

// class Solution {
// public:
//     vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
//         sort(people.begin(), people.end(), [&](vector<int>a, vector<int>b){
//             if(a[0] == b[0]) return a[1] < b[1];
//             return a[0] > b[0];
//         });
//         vector<vector<int>> res;
//         for(auto i : people){
//             res.insert(res.begin() + i[1], i);
//         }
//         return res;
//     }
// };

// 用链表的话效率更高一点
class Solution {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(), people.end(), [&](vector<int>a, vector<int>b){
            if(a[0] == b[0]) return a[1] < b[1];
            return a[0] > b[0];
        });

        list<vector<int>> lst;
        for (auto i : people){
            int pos = i[1];
            auto it = lst.begin();
            while (pos--) it++;
            lst.insert(it, i);
        }
        return vector<vector<int>>(lst.begin(), lst.end());
    }
};