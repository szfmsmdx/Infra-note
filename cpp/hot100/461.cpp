using namespace std;

class Solution {
public:
    int hammingDistance(int x, int y) {
        x = x ^ y;
        int res = 0;
        while (x) {
            res += x % 2;
            x /= 2;
        }
        return res;
    }
};