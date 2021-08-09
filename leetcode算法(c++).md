# leetcode学习

## 最长子串

如果我们依次递增地枚举子串的起始位置，那么子串的结束位置也是递增的！这里的原因在于，假设我们选择字符串中的第 k 个字符作为起始位置，并且得到了不包含重复字符的最长子串的结束位置为 r ~k~。那么当我们选择第 k+1个字符作为起始位置时，首先从 k+1 到 r~k~的字符显然是不重复的，并且由于少了原本的第 k 个字符，我们可以尝试继续增大 ，直到右侧出现了重复字符为止。这样一来，我们就可以使用==「滑动窗口」==来解决这个问题了：

==滑动窗口==

- 我们使用两个指针表示字符串中的某个子串（或窗口）的左右边界，其中左指针代表着上文中「枚举子串的起始位置」，而右指针即为上文中的 r~k~；
- 在每一步的操作中，我们会将左指针向右移动一格，表示 我们开始枚举下一个字符作为起始位置，然后我们可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着 以左指针开始的，不包含重复字符的最长子串。我们记录下这个子串的长度；
- 在枚举结束后，我们找到的最长的子串的长度即为答案。

判断重复字符

在上面的流程中，我们还需要使用一种数据结构来判断 是否有重复的字符，常用的数据结构为哈希集合（即 C++ 中的 std::unordered_set，Java 中的 HashSet，Python 中的 set, JavaScript 中的 Set）。在左指针向右移动的时候，我们从哈希集合中移除一个字符，在右指针向右移动的时候，我们往哈希集合中添加一个字符。

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        unordered_set<char> occ;
        int n = s.size();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        // 枚举左指针的位置，初始值隐性地表示为 -1
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1])) {
                // 不断地移动右指针
                // count方法在容器中查找值为 s[rk+1] 的元素的个数。
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1);
        }
        return ans;
    }
};
```

## 查找两个有序数组的中位数

为了降低时间、空间复杂度，采用二分查找

假设两个数组的长度分别是m和n

时间复杂度：O(log~2~(m+n))

空间复杂度：O(1)

```c++
class Solution {
public:
    int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;

        while (true) {
            // 边界情况
            if (index1 == m) {
                return nums2[index2 + k - 1];
            }
            if (index2 == n) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return min(nums1[index1], nums2[index2]);
            }

            // 正常情况
            int newIndex1 = min(index1 + k / 2 - 1, m - 1);
            int newIndex2 = min(index2 + k / 2 - 1, n - 1);
            int pivot1 = nums1[newIndex1];
            int pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            }
            else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2 == 1) {
            return getKthElement(nums1, nums2, (totalLength + 1) / 2);
        }
        else {
            return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        }
    }
};

```

## 最长回文子串

方法一：动态规划

对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。例如对于字符串 
“ababa”，如果我们已经知道 “bab” 是回文串，那么 “ababa” 一定是回文串，这是因为它的首尾两个字母都是 “a”。

根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 
P(i,j)表示字符串 s 的第 i 到 j 个字母组成的串（下文表示成 s[i:j]）是否为回文串，true,如果子串是回文串。false,其它情况。

这里的「其它情况」包含两种可能性：

1. s[i,j] 本身不是一个回文串；
2. i>j，此时 s[i,j] 本身不合法。

那么我们就可以写出动态规划的状态转移方程：
P(i,j)=P(i+1,j−1)∧(S~i~==S~j~)

也就是说，只有s[i+1:j−1] 是回文串，并且 s 的第 i 和 j 个字母相同时，s[i:j] 才会是回文串。

上文的所有讨论是建立在子串长度大于 2 的前提之上的，我们还需要考虑动态规划中的边界条件，即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，只要它的两个字母相同，它就是一个回文串。因此我们就可以写出动态规划的边界条件：P(i,i)=true;P(i,i+1)=(S~i~==S~i+1~) 根据这个思路，我们就可以完成动态规划了，最终的答案即为所有P(i,j)=true 中 
j−i+1（即子串长度）的最大值。注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。

```c++
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.size();
      //特殊情况
        if (n < 2) {
            return s;
        }
      //一般情况
        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 s[i..j] 是否是回文串
        vector<vector<int>> dp(n, vector<int>(n));
        // 初始化：所有长度为 1 的子串都是回文串
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        // 递推开始
        // 先枚举子串长度
        for (int L = 2; L <= n; L++) {
            // 枚举左边界，左边界的上限设置可以宽松一些
            for (int i = 0; i < n; i++) {
                // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                int j = L + i - 1;
                // 如果右边界越界，就可以退出当前循环
                if (j >= n) {
                    break;
                }

                if (s[i] != s[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 只要 dp[i][j]] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substr(begin, maxLen);
    }
};
```

时间复杂度：O(n^2)

空间复杂度: O(n^2)

方法二：中心扩展算法

方法一中的状态转移方程：

- P(i,i)=true
- P(i,i+1)=(S~i~==S~I+1~)
- P(i,j)=P(i+1,j-1)^(S~i~==S~I+1~)

找出其中的状态转移链：某一边界情况->........->P(i+1,j-1)->P(i,j)

可以发现，**所有的状态在转移的时候的可能性都是唯一的**。也就是说，我们可以从每一种边界情况开始「扩展」，也可以得出所有的状态对应的答案。

边界情况即为子串长度为 1 或 2 的情况。我们枚举每一种边界情况，并从对应的子串开始不断地向两边扩展。如果两边的字母相同，我们就可以继续扩展，例如从 P(i+1,j−1) 扩展到 P(i,j)；如果两边的字母不同，我们就可以停止扩展，因为在这之后的子串都不能是回文串了。

「边界情况」对应的子串实际上就是我们「扩展」出的回文串的「回文中心」。方法二的本质即为：我们枚举所有的「回文中心」并尝试「扩展」，直到无法扩展为止，此时的回文串长度即为此「回文中心」下的最长回文串长度。我们对所有的长度求出最大值，即可得到最终的答案。

```c++
class Solution {
public:
    pair<int, int> expandAroundCenter(const string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return {left + 1, right - 1};
    }

    string longestPalindrome(string s) {
        int start = 0, end = 0;
        for (int i = 0; i < s.size(); ++i) {
            auto [left1, right1] = expandAroundCenter(s, i, i);
            auto [left2, right2] = expandAroundCenter(s, i, i + 1);
            if (right1 - left1 > end - start) {
                start = left1;
                end = right1;
            }
            if (right2 - left2 > end - start) {
                start = left2;
                end = right2;
            }
        }
        return s.substr(start, end - start + 1);
    }
};
```

时间复杂度：O(n^2)，其中 n 是字符串的长度。长度为 1 和 2 的回文中心分别有 n 和 n−1 个，每个回文中心最多会向外扩展 O(n) 次。

空间复杂度：O(1)

## Z字排序

自己写的：用二维数组实现的按行排序

```c++
class Solution {
public:
    //Z字变换
    string convert(string s, int numRows) {
        //边界情况
        if(numRows==1){return s;}
        int n=s.length();
        char dp[1000][1000];
        string res;
        //dp表示最终的二维数组
        int line =0;
        int col=0;
        for (int i = 0; i < n; ++i) {
            int gro=(i)/(numRows*2-2);
            if((i+1)%(2*numRows-2)>numRows){
                line=2*numRows-(i+1)%(2*numRows-2);
                col=gro*(numRows-1)+(i+1)%(2*numRows-2)%numRows;
            }
            else if((i+1)%(2*numRows-2)==0 ){
                line=2;
                col=(gro+1)*(numRows-1)-1;
            }
            else {
                line=(i+1)%(2*numRows-2);
                col=gro*(numRows-1);
            }
            dp[line][col]=s[i];
        }
//        cout<<n*(numRows-1)/(2*numRows-2)+1<<endl;
        for(int i=1;i<=numRows;i++){
            for(int j=1;j<=n*(numRows-1)/(2*numRows-2)+1;j++){
                if(dp[i][j-1]!='\0'){res+=dp[i][j-1];dp[i][j-1]='\0';}
            }

        }
        return res;
    }
};
```

时间复杂度：O(n^2+n)

空间复杂度：O(1000000)

方法一：按行排序

思路：通过从左向右迭代字符串，我们可以轻松地确定字符位于 Z 字形图案中的哪一行。

算法：我们可以使用 min(numRows,len(s)) 个列表来表示 Z 字形图案中的非空行。从左到右迭代 s，将每个字符添加到合适的行。可以使用当前行和当前方向这两个变量对合适的行进行跟踪。

只有当我们向上移动到最上面的行或向下移动到最下面的行时，当前方向才会发生改变。

```c++
class Solution {
public:
    string convert(string s, int numRows) {

        if (numRows == 1) return s;

        vector<string> rows(min(numRows, int(s.size())));
        int curRow = 0;
        bool goingDown = false;

        for (char c : s) {
            rows[curRow] += c;
            if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }

        string ret;
        for (string row : rows) ret += row;
        return ret;
    }
};
```

时间复杂度：O(n)

空间复杂度：O(n)



方法二：按行访问

思路：按照与逐行读取 Z 字形图案相同的顺序访问字符串。

算法：首先访问 行 0 中的所有字符，接着访问 行 1，然后 行 2，依此类推...

对于所有整数 k，行 0 中的字符位于索引 k(2⋅numRows−2) 处;行 numRows−1 中的字符位于索引k(2⋅numRows−2)+numRows−1 处;内部的 行 i 中的字符位于索引 k(2⋅numRows−2)+i 以及 (k+1)(2⋅numRows−2)−i 处;

```c++
class Solution {
public:
    string convert(string s, int numRows) {

        if (numRows == 1) return s;

        string ret;
        int n = s.size();
        int cycleLen = 2 * numRows - 2;

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j + i < n; j += cycleLen) {
                ret += s[j + i];
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
                    ret += s[j + cycleLen - i];
            }
        }
        return ret;
    }
};
```

时间复杂度：O(n)

空间复杂度：O(n)

## 整数反转

 自己写的：

```c++
class Solution {
public:
    //整数反转
    int reverse(int x) {
        if(x<-2147483648||x>2147483647||x==0){
            return 0;
        }
        bool isneg= false;
        if(x<0) isneg = true;
        long temp= abs(x);
        long res=0;
        while (temp/10>0){
            if((temp%10+res*10)<-2147483648||(temp%10+res*10)>2147483647){
                return 0;
            }
            res=temp%10+res*10;
            temp=temp/10;
        }
        if((temp%10+res*10)<-2147483648||(temp%10+res*10)>2147483647){
            return 0;
        }
        res=res*10+temp;
        return isneg?-res:res;

    }
};
```



```c++
class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            if (rev < INT_MIN / 10 || rev > INT_MAX / 10) {
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }
};
```

时间复杂度：O(lgx)   x的位数

空间复杂度：O(1)

两个代码时空间复杂度一样，答案给的代码更加简洁

## 字符串转换整数

自己写的：

```c++
class Solution {
public:
    int myAtoi(string s) {
        string dig="0123456789";
        int value=0;
        int n=s.length();
        bool start=false,isdig=false,isneg=false;
        long res=0;
        for(int i=0;i<n;i++){
            isdig=false;
            for(int j=0;j<dig.length();j++){
                if(s[i]==dig[j]){
                    isdig=true;
                    value=j;
                    break;
                }
            }
            if(s[i]==' '){
                if(!start)continue;
                else break;
            }
            else if(s[i]=='+'||s[i]=='-'){
                if(start== true)break;
                start=true;
                isneg=s[i]=='-'?true:false;
                continue;
            }
            else if(isdig){
                if(start==false)start=true;

                if(isneg){
                    if(res>2147483648) {
                        res = 2147483648;
                        break;
                    } else{
                        res=res*10+value;
                    }
                }
                else{
                    if(res>2147483647){
                        res=2147483647;
                        break;
                    } else{
                        res=res*10+value;
                    }
                }




            }
            else{
                break;
            }

        }
        if(isneg){
            res=-res;
            if(res<-2147483648){
                res=-2147483648;
            }
        } else{
            if(res>2147483647){
                res=2147483647;
            }
        }
        return res;
    }
};
```

时间复杂度：O(n)

空间复杂度：O(1)

答案方法：自动机

字符串处理的题目往往涉及复杂的流程以及条件情况，如果直接上手写程序，一不小心就会写出极其臃肿的代码。

因此，为了有条理地分析每个输入字符的处理方法，我们可以使用自动机这个概念：

我们的程序在每个时刻有一个状态 s，每次从序列中输入一个字符 c，并根据字符 c 转移到下一个状态 s'。这样，我们只需要建立一个覆盖所有情况的从 s 与 c 映射到 s' 的表格即可解决题目中的问题。

本题自动机：

![](http://qungz.photo.store.qq.com/qun-qungz/V54FnPGr3TZMJ30yuFaX3gugmm1dm5Bt/V5bCQAyNDcyODgxNDIw7gxhhWaxFQ!!/800?w5=995&h5=800&rf=viewer_421&t=5)

接下来编程部分就非常简单了：我们只需要把上面这个状态转换表抄进代码即可。

另外自动机也需要记录当前已经输入的数字，只要在 s' 为 in_number 时，更新我们输入的数字，即可最终得到输入的数字。

```c++
class Automaton {
    string state = "start";
    unordered_map<string, vector<string>> table = {
        {"start", {"start", "signed", "in_number", "end"}},
        {"signed", {"end", "end", "in_number", "end"}},
        {"in_number", {"end", "end", "in_number", "end"}},
        {"end", {"end", "end", "end", "end"}}
    };

    int get_col(char c) {
        if (isspace(c)) return 0;
        if (c == '+' or c == '-') return 1;
        if (isdigit(c)) return 2;
        return 3;
    }
public:
    int sign = 1;
    long long ans = 0;

    void get(char c) {
        state = table[state][get_col(c)];
        if (state == "in_number") {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? min(ans, (long long)INT_MAX) : min(ans, -(long long)INT_MIN);
        }
        else if (state == "signed")
            sign = c == '+' ? 1 : -1;
    }
};

class Solution {
public:
    int myAtoi(string str) {
        Automaton automaton;
        for (char c : str)
            automaton.get(c);
        return automaton.sign * automaton.ans;
    }
};
```

时间复杂度：O(n),其中 *n* 为字符串的长度。我们只需要依次处理所有的字符，处理每个字符需要的时间为*O*(1)。

空间复杂度：O(1)，自动机的状态只需要常数空间存储。

# 回文数

自己写的

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        vector<int> dp;
        int temp=x;
        if(x<0)return false;
        while(temp>0){
            dp.push_back((temp%10));
            temp=temp/10;
        }
        int n=dp.size();
        for(int i=0;i<n;i++){
            if(dp[i]!=dp[n-i-1]){
                return false;
            }
        }
        return true;
    }
};
```

时间复杂度：O(lgn)

空间复杂度：O(lgn)

答案方法：反转一半数字

**思路：**

映入脑海的第一个想法是将数字转换为字符串，并检查字符串是否为回文。但是，这需要额外的非常量空间来创建问题描述中所不允许的字符串。

第二个想法是将数字本身反转，然后将反转后的数字与原始数字进行比较，如果它们是相同的，那么这个数字就是回文。
但是，如果反转后的数字大于 int.MAX，我们将遇到整数溢出问题。

按照第二个想法，为了避免数字反转可能导致的溢出问题，为什么不考虑只反转 int 数字的一半？毕竟，如果该数字是回文，其后半部分反转后应该与原始数字的前半部分相同。

例如，输入 1221，我们可以将数字 “1221” 的后半部分从 “21” 反转为 “12”，并将其与前半部分 “12” 进行比较，因为二者相同，我们得知数字 1221 是回文。

**算法：**

首先，我们应该处理一些临界情况。所有负数都不可能是回文，例如：-123 不是回文，因为 - 不等于 3。所以我们可以对所有负数返回 false。除了 0 以外，所有个位是 0 的数字不可能是回文，因为最高位不等于 0。所以我们可以对所有大于 0 且个位是 0 的数字返回 false。

现在，让我们来考虑如何反转后半部分的数字。

对于数字 1221，如果执行 1221 % 10，我们将得到最后一位数字 1，要得到倒数第二位数字，我们可以先通过除以 10 把最后一位数字从 1221 中移除，1221 / 10 = 122，再求出上一步结果除以 10 的余数，122 % 10 = 2，就可以得到倒数第二位数字。如果我们把最后一位数字乘以 10，再加上倒数第二位数字，1 * 10 + 2 = 12，就得到了我们想要的反转后的数字。如果继续这个过程，我们将得到更多位数的反转数字。

现在的问题是，我们如何知道反转数字的位数已经达到原始数字位数的一半？

==由于整个过程我们不断将原始数字除以 10，然后给反转后的数字乘上 10，所以，当原始数字小于或等于反转后的数字时，就意味着我们已经处理了一半位数的数字了。==

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        // 特殊情况：
        // 如上所述，当 x < 0 时，x 不是回文数。
        // 同样地，如果数字的最后一位是 0，为了使该数字为回文，
        // 则其第一位数字也应该是 0
        // 只有 0 满足这一属性
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }

        int revertedNumber = 0;
        while (x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }

        // 当数字长度为奇数时，我们可以通过 revertedNumber/10 去除处于中位的数字。
        // 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
        // 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
        return x == revertedNumber || x == revertedNumber / 10;
    }
};
```

时间复杂度：O(lgn)

空间复杂度：O(1)

## 正则表达式匹配

方法一：动态规划

![](http://qungz.photo.store.qq.com/qun-qungz/V54FnPGr3TZMJ30yuFaX3gugmm1dm5Bt/V5bCQAyNDcyODgxNDKEkBBhaFpSHw!!/800?w5=1384&h5=430&rf=viewer_421&t=5)

```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i, int j) {
            if (i == 0) {
                return false;
            }
            if (p[j - 1] == '.') {
                return true;
            }
            return s[i - 1] == p[j - 1];
        };

        vector<vector<int>> f(m + 1, vector<int>(n + 1));
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    f[i][j] |= f[i][j - 2];
                    if (matches(i, j - 1)) {
                        f[i][j] |= f[i - 1][j];
                    }
                }
                else {
                    if (matches(i, j)) {
                        f[i][j] |= f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }
};

```

