## 常见问题汇总

new 和 delete要连着用。

nums.size()是无符号数，要先len  = nums.size(), 再len - 1. 如果减完小于0，会溢出。

如果需要和数组下一位比较，有需要可以在数组末尾加一个无关的元素。见 无聊的题目 - 38

## 需要留意的题目

33 237 

## 链表

### Pass

21 

### 数据结构

```c++
struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};
```

### 2. Add Two Numbers

#### 题目描述

​		两个链表，返回一个新的链表，每个值是输入链表对应的和

#### 解法

​		遍历，注意进位，注意某个链表已经走到头

```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p = l1, *q = l2, *dummy = new ListNode(-1), *pre = dummy;
        int flag = 0, num;
        while(p && q){
            num = (flag + p->val + q->val) % 10;
            flag = (flag + p->val + q->val) / 10;
            p = p->next;
            q = q->next;
            ListNode *tmp = new ListNode(num);
            pre->next = tmp;
            pre = pre->next;
        }
        while(p){
            num = (flag + p->val) % 10;
            flag = (flag + p->val) / 10;
            p = p->next;
            ListNode *tmp = new ListNode(num);
            pre->next = tmp;
            pre = pre->next;
        }
        while(q){
            num = (flag + q->val) % 10;
            flag = (flag + q->val) / 10;
            q = q->next;
            ListNode *tmp = new ListNode(num);
            pre->next = tmp;
            pre = pre->next;
        }
        if(flag){
            ListNode *tmp = new ListNode(1);
            pre->next = tmp;
        }
        return dummy->next;
    }
};
```



### 19. Remove Nth Node From End of List

#### 题目描述

​		删掉倒数第几个链表节点

#### 解法

​		两个指针，一个先走k步。然后两个指针同时走，第一个指针走到最后一个点，将第二个指针的next删除即可。

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *p = dummy, *q = dummy;
        for(int i=0; i<n; i++)
            p = p->next;
        while(p->next){
            p = p->next;
            q = q->next;
        }
        ListNode *tmp = q->next;
        q->next = q->next->next;
        delete tmp;
        return dummy->next;
    }
};
```

### * 24. Swap Nodes in Pairs

#### 题目描述

​		两两节点交换。

#### 解法

​		使用pre节点，指向下一组（两个节点，或只剩一个）节点的第二个。然后两两交换，直到不需要交换为止。

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *first = head, *second, *dummy = new ListNode(-1), *pre = dummy;
        while(1){
            if(!first || !first->next){
                pre->next = first;
                break;
            }
            second = first->next;
            pre->next = second;
            first->next = second->next;
            second->next = first;
            pre = first;
            
            first = first->next;
        }
        return dummy->next;
    }
};
```



### * 61. Rotate List

#### 题目描述

​		将链表旋转

#### 解法

​		k可能很大，需要先取模。两种办法，第一种是考虑需要找到倒数第k % len个节点（从后面看），一种是(len - k % len) % len从前面看。

``` c++
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        int len = 0;
        ListNode* now = head;
        for(; now; now=now->next)
            len ++;
        if(!len)
            return NULL;
        k = (len - k % len) % len;
        if(!k)
            return head;
        now = head;
        for(int i=0; i<k-1; i++)
            now = now->next;
        ListNode *ans = now->next;
        now->next=NULL;
        for(now=ans; now->next!=NULL; now=now->next);
        now->next= head;
        return ans;
    }
};
```



### 83. Remove Duplicates from Sorted List

#### 题目描述

​		删除重复节点

#### 解法 

​		一次遍历，节点的值和上一个不一样，这个节点不删。

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head)
            return NULL;
        ListNode *pre = head, *now = head->next; 
        for(; now; now=now->next){
            if(now->val!=pre->val)
                pre = now;
            else
                pre->next = now->next;
        }
        return head;
    }
};
```

### 160. Intersection of Two Linked Lists

#### 题目描述

​		求两个链表的交点，如无交点，返回空指针。

#### 解法

​		两个指针分别从两个链表开头遍历，若走到某个位置两个指针相等，则退出。如果在退出之前，走到空指针，下一步从空指针到另一个链表的头。

``` c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *p = headA, *q = headB;
        while(p != q){
            if(!p)
                p = headB;
            else
                p = p->next;
            if(!q)
                q = headA;
            else
                q = q->next;
        }
        return p;
    }
};
```

#### 注意

​		为什么不能走到空指针，本步骤直接跳到下一个指针呢？

​		因为两个链表可能没有交点，走到空指针需要等待一下，等到另一个节点走完当前步，判断两个点是不是都为空，如果都为空，则没有交点，如果不都为空，则继续往前走，等待退出条件。

### 206. Reverse Linked List

#### 题目描述

​		反转链表

#### 解法

``` c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *dummy = new ListNode(-1);
        for(ListNode *now=head; now!=NULL; ){
            ListNode *tmp = now->next;
            now->next = dummy->next;
            dummy->next = now;
            now = tmp;
        }
        return dummy->next;
    }
};
```



### * 237. Delete Node in a Linked List 

#### 题目描述

​		删除给定节点

#### 解法

​		将下一个点的值复制到本节点，再删除当前节点，骚啊。

``` c++
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```



## 树

### 树的数据结构

注意某些时候有father节点。

```c++
Definition for a binary tree node.
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
```

### 94.  Binary Tree Inorder Traversal

#### 题目描述

​		中序遍历二叉树，返回一个vector

#### 解法

​		非递归，使用栈

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode *> s;
        TreeNode *now = root;
        while(now){
            s.push(now);
            now = now->left;
        }
        while(!s.empty()){
            now = s.top();
            ans.push_back(now->val);
            s.pop();

            now = now->right;
            while(now){
                s.push(now);
                now = now->left;
            }
        }
        return ans;
    }
};
```



### 98.  Validate Binary Search Tree

#### 题目描述

​		验证一个树是不是二叉搜索树，注意，某节点和左右子节点相等也不符合要求。

#### 解法

​		递归，对每一个节点，有一个合法取值的范围，不在这个范围之内，即为不符合要求。

```c++
class Solution {
public:
    bool check(TreeNode *node, long mx, long mn){
        if(!node)
            return true;
        if(node->val >= mx || node->val<=mn)
            return false;
        return check(node->left, node->val, mn) && check(node->right, mx, node->val);
    }
    bool isValidBST(TreeNode* root) {
        return check(root, LONG_MAX, LONG_MIN);
    }
};
```

#### 注意

​		可能数据会卡INT_MAX，所以可以将一开始范围限定在LONG_MAX.

### 101.  Symmetric Tree

#### 题目描述

​		判断一棵树是否为左右对称

#### 解法

​		递归

``` c++
class Solution {
public:
    bool check(TreeNode *p, TreeNode *q){
        if(!p && !q) 
            return true;
        if(!p || !q)
            return false;
        if(p->val != q->val)
            return false;
        return check(p->left, q->right) && check(p->right, q->left);
    }
    bool isSymmetric(TreeNode* root) {
        if(!root)
            return true;
        return check(root->left, root->right);
    }
};
```



### 102. Binary Tree Level Order Traversal

#### 题目描述

​		分层打印二叉树

#### 解法

​		两个队列，分别存节点和当前节点的深度。当前节点深度大于之前节点的最大深度时，将单层节点存进答案，将单层节点数组tmp清空，跳出循环之后，也要将最后一层的节点存进答案。

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int> > ans;
        vector<int> tmp;
        queue<TreeNode*> q;
        queue<int> deep;
        if(!root)
            return ans;
        q.push(root);
        deep.push(1);
        int now_max = 1;
        while(!q.empty()){
            int now_deep = deep.front();
            TreeNode *now = q.front();
            q.pop();
            deep.pop();
            
            if(now_deep > now_max){
                now_max ++;
                ans.push_back(tmp);
                vector<int>().swap(tmp);
            }
            tmp.push_back(now->val);
            if(now->left){
                q.push(now->left);
                deep.push(now_deep + 1);
            }
            if(now->right){
                q.push(now->right);
                deep.push(now_deep + 1);
            }
        }
        ans.push_back(tmp);
        return ans;
    }
};
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

#### 题目描述

​		根据中序遍历，先序遍历的结果构建树

#### 解法

​		先序遍历的第一个点一定是根节点，在中序遍历序列找到这个值，他的左边是左子树，右边是右子树。根据这个将树不断往下分。

```  c++
class Solution {
public:
    unordered_map<int, int> hash;
    TreeNode* dfs(vector<int>& pre, vector<int>& in, int l1, int r1, int l2, int r2){
        // 注意边界条件的判断
        if(l1 > r1)
            return NULL;
        TreeNode *node = new TreeNode(pre[l1]);

        int len = hash[pre[l1]] - l2;
        node->left = dfs(pre, in, l1 + 1, l1 + len, l2, l2 + len - 1);
        node->right = dfs(pre, in, l1 + len + 1, r1, l2 + len + 1, r2);

        return node;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for(int i=0; i<inorder.size(); i++)
            hash[inorder[i]] = i;
        return dfs(preorder, inorder, 0, preorder.size() - 1, 0, inorder.size() - 1);
    }
};
```



#### 注意

​		注意返回条件，当l>r的时候，就是空了。

### 124. Binary Tree Maximum Path Sum

#### 题目描述

​		给一个二叉树，返回这个树任意路径上数字的最大的和，不必经过根节点，不必到达叶子节点

#### 解法

​		递归，对每一个点，求一个以他为最上面节点的路径和。

``` c++
class Solution {
public:
    int dfs(TreeNode* node, int& ans){
        if(!node)
            return 0;
        int left = max(0, dfs(node->left, ans));
        int right = max(0, dfs(node->right, ans));
        
        ans = max(ans,left + right + node->val);
        return max(left, right) + node->val;
    }
    int maxPathSum(TreeNode* root) {
        if(!root)
            return 0;
        int ans = root->val;
        dfs(root, ans);
        return ans;
    }
};
```

#### 注意

​		注意left right如果是负数，则不加上。

### 173. Binary Search Tree Iterator

#### 题目描述

​		很无聊的题目，将中序遍历拆成接口

#### 解法

​		非递归的中序遍历

```c++
class BSTIterator {
public:
    stack<TreeNode *> s;
    BSTIterator(TreeNode* root) {
        TreeNode *now = root;
        while(now){
            s.push(now);
            now = now->left;
        }
    }
    
    /** @return the next smallest number */
    int next() {
        TreeNode *now = s.top();
        int ans = now->val;
        s.pop();
        now = now->right;
        while(now){
            s.push(now);
            now = now->left;
        }
        return ans;
    }
    
    /** @return whether we have a next smallest number */
    bool hasNext() {
        if(s.empty())
            return false;
        return true;
    }
};
```



### 236.  Lowest Common Ancestor of a Binary Tree

#### 题目描述

​		找到两个节点的最低公共祖先

#### 解法

​		从根节点进行递归，对于每一个节点返回左右子节点是否含有目标要求的p/q节点。如果left + right + now == 2，说明这个点处是公共祖先。且由于递归的作用，搜索的位置是叶子节点到根节点的，满足最低公共祖先的要求。当找到这样的祖先之后，不可能left + right + now == 2了，因为这个子节点只返回1，无法加到2.

``` c++
class Solution {
public:
    int dfs(TreeNode* node, TreeNode* p, TreeNode* q, TreeNode*& ans){
        if(!node)
            return 0;
        int left = dfs(node->left, p, q, ans);
        int right = dfs(node->right, p, q, ans);
        
        int now = (node->val == p->val || node->val == q->val);
        if(now + left + right == 2)
            ans = node;
        return now + left + right > 0? 1: 0;
            
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode *ans = NULL;
        dfs(root, p, q, ans);
        return ans;
    }
};
```

### 297. Serialize and Deserialize Binary Tree

#### 题目描述

​		对二叉树先字符串编码，再解码

#### 解法

​		无聊的一匹，注意负号。

```c++
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string s = "[";
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            TreeNode *tmp = q.front();
            q.pop();
            if(!tmp){
                s += "null,";
                continue;
            }
            else
                s += to_string(tmp->val) + ',';
            q.push(tmp->left);
            q.push(tmp->right);
        }
        s += ']';
        return s;
    }
    int convert(string &data, int &i){
        int len = data.size() - 1;
        int flag = 1, num = 0;
        if(data[i] == '-'){
            flag = -1;
            i++;
        }
        while(i<len && data[i]>='0' && data[i]<='9'){
            num = num * 10 + (data[i] - '0');
            i ++;
        }
        num *= flag;
        i++;
        return num;
    }
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        int i = 1;
        if(data[i] == 'n')
            return NULL;
        queue<TreeNode*> q;
        int len = data.size() - 1;
        int num = convert(data, i);
        TreeNode *root = new TreeNode(num), *now;
        q.push(root);
        while(!q.empty()){
            now = q.front();
            q.pop();
            if(data[i] == 'n'){
                i += 5;
            }
            else{
                num = convert(data, i);
                now->left = new TreeNode(num);
                q.push(now->left);
            }
            if(data[i] == 'n'){
                i += 5;
            }
            else{
                num = convert(data, i);
                now->right = new TreeNode(num);
                q.push(now->right);
            }
        }
        return root;
    }
};
```

#### 注意

​		注意负号！！负号！！

### 543. Diameter of Binary Tree

#### 题目描述

​		求二叉树中，任意两个节点的最长距离

#### 解法

​		递归求解，对每一个节点，求一个以它为最上面点的（“根节点”）的路径长度，遇见更大的距离，则刷新。

```c++
class Solution {
public:
    int dfs(TreeNode* node, int &ans){
        if(!node)
            return 0;
        int left = dfs(node->left, ans);
        int right = dfs(node->right, ans);
        int now = left + right;
        
        ans = max(now, ans);
        return max(left, right) + 1;
    }
    
    int diameterOfBinaryTree(TreeNode* root) {
        if(!root)
            return 0;
        int ans = 0;
        dfs(root, ans);
        return ans;
    }
};
```

## 贪心

###　11. Container With Most Water

#### 题目描述

​		有一排高度，求能存水最多的量。（与最大矩形不同）

####　解法

​		贪心法，双指针分别指向数组头尾，指向更小数的指针向中间缩。

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size() - 1, ans = 0;
        while(l < r){
            ans = max(ans, (r - l) * min(height[l], height[r]));
            if(height[l] <= height[r])
                l ++;
            else
                r --;
        }
        return ans;
    }
};
```

## 字符串



## 递推

### 55. Jump Game

#### 题目描述

​		能不能走到台阶最后一位

#### 解法

​		递推

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int reach = 0;
        for(int i=0; i<=reach; i++){
            reach = max(nums[i] + i, reach);
            if(reach >= nums.size() - 1)
                return true;
        }
        return false;
    }
};
```





## 动态规划

### * 10. Regular Expression Matching

#### 题目描述

​		正则表达式，含有. *两种匹配符。

#### 解法 

​		dp深搜+记忆化搜索

```c++
class Solution {
public:
    vector<vector<int> > match;
    bool check(string s, string p, int x, int y, int l1, int l2){
        if(match[x][y] != -1)
            return match[x][y];
        if(y == l2){
            if(x == l1)
                return true;
            match[x][y] = false;
            return match[x][y];
        }
        bool now = x < l1 && (s[x] == p[y] || p[y] == '.');
        if(y+1<l2 && p[y+1]=='*')
            match[x][y] = (now && check(s, p, x+1, y, l1, l2)) || check(s, p, x, y+2, l1, l2);
        else
            match[x][y] = now && check(s, p, x+1, y+1, l1, l2);
        return match[x][y];
    }
    bool isMatch(string s, string p) {
        int l1 = s.size(), l2 = p.size();
        match = vector<vector<int> >(l1+1, vector<int>(l2+1, -1));
        return check(s, p, 0, 0, l1, l2);   
    }
};
```

### 53. Maximum Subarray

#### 题目描述

​		和最大的连续子数组

#### 解法

​		一遍遍历，每次求出当前位置的最大值，更新答案

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = nums[0], now = nums[0];
        for(int i=1; i<nums.size(); i++){
            now = max(now + nums[i], nums[i]);
            ans = max(ans, now);
        }
        return ans;
    }
};
```

### 62. Unique Paths

#### 题目描述

​		dp问题

#### 解法

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int> > dp(m + 1, vector<int>(n + 1, 0));
        for(int i=1; i<=m; i++){
            for(int j=1; j<=n; j++)
                if(i==j && j==1)
                    dp[i][j] = 1;
                else
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
        return dp[m][n];
    }
};
```







## 搜索

###　17. Letter Combinations of a Phone Number

####　题目描述

​		电话上的键盘转字符串，深搜。

####　解法

```c++
class Solution {
public:
    unordered_map<int, string> hash;
    void dfs(string num, vector<string>& ans, string tmp, int k, int len){
        if(k == len){
            ans.push_back(tmp);
            return;
        }
        for(int i=0; i<hash[num[k]-'0'].size(); i++)
            dfs(num, ans, tmp + hash[num[k]-'0'][i], k+1, len);
    }
    
    vector<string> letterCombinations(string digits) {
        vector<string> ans;
        if(!digits.size()) 
            return ans;
        hash[2] = "abc";
        hash[3] = "def";
        hash[4] = "ghi";
        hash[5] = "jkl";
        hash[6] = "mno";
        hash[7] = "pqrs";
        hash[8] = "tuv";
        hash[9] = "wxyz";
        string tmp = "";
        int len = digits.size();
        dfs(digits, ans, tmp, 0, digits.size());
        return ans;
    }
};
```

### 22. Generate Parentheses

#### 题目描述

​		输出所有符合要求的括号序列

#### 解法

​		深搜

```c++
class Solution {
public:
    void dfs(vector<string>& ans, string tmp, int l, int r, int n){
        if(l == r && l == n){
            ans.push_back(tmp);
            return;
        }
        if(l < n)
            dfs(ans, tmp + '(', l + 1, r, n);
        if(l > r)
            dfs(ans, tmp + ')', l, r + 1, n);
            
        
    }
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        string tmp = "";
        dfs(ans, tmp, 0, 0, n);
        return ans;
    }
};
```

### 46. Permutations

#### 题目描述

​		数字排列 无重复

#### 解法

​		深搜

```c++
class Solution {
public:
    vector<bool> visit;
    void dfs(vector<int> &nums, vector<vector<int> > &ans, vector<int> tmp, int k, int n){
        if(k==n){
            ans.push_back(tmp);
            return;
        }    
        // unordered_map<int, int> hash;
        for(int i=0; i<nums.size(); i++){
            // if(visit[i] || hash.find(nums[i]) != hash.end())
            if(visit[i])
                continue;
            visit[i] = true;
            // hash[nums[i]] = 1;
            tmp.push_back(nums[i]);
            dfs(nums, ans, tmp, k + 1, n);
            visit[i] = false;
            tmp.pop_back();
        }
    }
    
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int> > ans;
        vector<int> tmp;
        visit = vector<bool>(nums.size(), false);
        dfs(nums, ans, tmp, 0, nums.size());
        return ans;
    }
};
```



## 二分

### 23. Merge k Sorted Lists

#### 题目描述

​		对k个排序链表做归并

#### 解法

​		二分归并

```c++
class Solution {
public:
    ListNode* merge(vector<ListNode*>& lists, int l, int r){
        if(l >= r)
            return lists[l];
        int mid = (l + r) / 2;
        ListNode *left = merge(lists, l, mid);
        ListNode *right = merge(lists, mid + 1, r);
        
        ListNode *p = left, *q = right, *dummy = new ListNode(-1), *pre = dummy;
        while(p && q){
            if(p->val <= q->val){
                pre->next = p;
                p = p->next;
                pre = pre->next;
            }
            else{
                pre->next = q;
                q = q->next;
                pre = pre->next;
            }
        }
        while(p){
            pre->next = p;
            p = p->next;
            pre = pre->next;
        }
        while(q){
            pre->next = q;
            q = q->next;
            pre = pre->next;
        }
        return dummy->next;
    }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(!lists.size())
            return NULL;
        return merge(lists, 0, lists.size()-1);
    }
};
```



### 69. sqrt(x)

#### 题目描述

​		求平方根，输入8，输出2，这个也是比较容易错的用例。

#### 解法 

​		二分的过程为逐渐缩小答案的过程，如果mid平方大于目标值，则缩小范围，将右值r减小。如果mid平方小于目标值，则不能增加左断点l。因为l+1可能就不符合要求了。

``` c++
class Solution {
public:
    int mySqrt(int x) {
        int l = 0, r = x, mid;
        while(l < r){
            mid = (l + r + 1ll) / 2;
            if(mid == x / mid)
                return mid;
            else if(mid > x / mid)
                r = mid - 1;
            else 
                l = mid;
        }
        return l;
    }
};
```

#### 注意

​		二分的时候，注意中点的取法，因为有可能当两个数字的时候，中点永远为左侧，而左侧符合要求，造成死循环。而加了1可能会int越界，故需要+1ll（long long）

### 34. Find First and Last Position of Element in Sorted Array

####　题目描述

​		一个排序数组，返回一个数字的第一次/最后一次出现的位置

#### 解法

​		二分

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1, mid, mmid;
        while(l<=r){
            mid = (l + r) / 2;
            if(nums[mid] == target)
                break;
            if(nums[mid] > target)
                r = mid - 1;
            else
                l = mid + 1;
        }
        if(l > r)
            return vector<int>{-1, -1};
        l = 0, r = mid;
        while(l < r){
            mmid = (l + r) / 2;
            if(nums[mmid] < target)
                l = mmid + 1;
            else
                r = mmid;
        }
        vector<int> ans;
        ans.push_back(l);
        l = mid, r = nums.size() - 1;
        while(l < r){
            mmid = (l + r + 1) / 2;
            if(nums[mmid] > target)
                r = mmid - 1;
            else
                l = mmid;
        }
        ans.push_back(l);
        return ans;
    }
};
```



### 35. Search Insert Position

#### 题目描述

​		从有序数组中找一个数，如果没找到，返回这个数在数组中应该插入的位置

#### 解法

​		二分，缩小范围的策略是，扔掉比这个数字小的数。只剩一个数字的时候，判断它是大于等于目标值，还是小小于目标值。

``` c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1, mid;
        while(l<r){
            mid = (l + r) / 2;
            if(nums[mid] == target)
                return mid;
            else if(nums[mid] < target)
                l = mid + 1;
            else
                r = mid;
        }
        if(nums[l]>=target)
            return l;
        return l + 1;
    }
};
```

#### 注意

​		只含一个数的时候，数组可能永远满足条件，会造成死循环，所以条件为l < r。

​		与上一个题不一样，这次我们是保留搜索范围右半部分，所以求mid不需要+1，在剩两个数字的时候，可能会出现区间不再缩小，会超时。如果mid = (l + r) / 2与l = mid一起用，或者mid = (l + r + 1)/2, r = mid一起用的话，就可能会引起超时。 

### 153. Find Minimum in Rotated Sorted Array

#### 题目描述

​		一个有序数组发生了旋转，找到最小的元素

#### 解法

​		如果数据单调，则直接返回左端点。如果不单调，看中点是否大于左端点。若大于等于，则说明最小值还在右面，l = mid + 1；若小于，则最小值在左面，r = mid。

``` c++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        while(l < r){
            if(nums[l] < nums[r])
                return nums[l];
            int mid = (l + r) / 2;
            if(nums[mid]>=nums[l])
                l = mid + 1;
            else
                r = mid;
        }
        return nums[l];
    }
};
```



### *** 33. Search in Rotated Sorted Array

#### 题目描述		

​		在一个旋转数组中找一个目标值

#### 解法

​		这题做了好几次了，结果还是不清楚。核心思想是先确定单调的部分（即中点的位置），然后再通过条件缩小范围。

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1, mid;
        while(l <= r){
            mid = (l + r) / 2;
            if(nums[mid] == target)
                break;
            else if(nums[mid] >= nums[l]){
                if(nums[mid] > target && nums[l] <= target)
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            else{
                if(nums[mid] < target && nums[r] >= target)
                    l = mid + 1;
                else
                    r = mid - 1;
            }
        }
        if(l <= r)
            return mid;
        return -1;
    }
};
```



## 栈

### 20. Valid Parentheses

#### 题目描述

​		验证括号序列是否合法

#### 解法

​		栈

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        for(int i=0; i<s.size(); i++){
            if(s[i] == '[' || s[i] == '(' || s[i] == '{')
                st.push(s[i]);
            else if(s[i] == ']' && (st.empty() || st.top() != '['))
                return false;
            else if(s[i] == '}' && (st.empty() || st.top() != '{'))
                return false;
            else if(s[i] == ')' && (st.empty() || st.top() != '('))
                return false;
            else
                st.pop();
        }
        return st.empty();
    }
};
```

### * 42. Trapping Rain Water

#### 题目描述

​		有一堆矩形，然后求能存多少水

#### 解法

​		单调栈

```c++
class Solution {
public:
    int trap(vector<int>& height) {
        stack<int> s;
        int ans = 0, top;
        for(int i=0; i<height.size(); i++){
            while(!s.empty() && height[s.top()] < height[i]){
                int top = s.top();
                s.pop();
                if(s.empty())
                    break;
                ans += (i - s.top() - 1) * (min(height[i], height[s.top()]) - height[top]);
            }
            s.push(i);
        }
        return ans;
    }
};
```



## 哈希

### 1. Two Sum 

求满足目标和数字的两个下标

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int len = nums.size();
        vector<int> ans;
        unordered_map<int, int> pos;
        for(int i=0; i<len; i++){
            if(pos.find(target - nums[i]) != pos.end()){
                ans.push_back(pos[target - nums[i]]);
                ans.push_back(i);
                break;
            }
            pos[nums[i]] = i;
        }
        return ans;
    }
};
```

### 3. Longest Substring Without Repeating Characters

#### 题目介绍

​		求最长的不含重复字符的连续子序列

#### 解法

​		使用哈希表，记录上一次出现某字符的位置。新来一个字符，如果之前已经出现过，则判断上一次的位置和当前合法的start下标哪个更大，更新start下标，然后求一个以当前字符为结尾的最长连续序列长度

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<int, int> pos;
        int start = -1, ans = 0;
        for(int i=0; i<s.size(); i++){
            if(pos.find(s[i]) != pos.end())
                start = max(start, pos[s[i]]);
            ans = max(ans, i - start);
            pos[s[i]] = i;
        }
        return ans;
    }
};
```

## 区间合并

### ** 56. Merge Intervals

#### 题目描述

​		给一堆起止点的区间，求合并之后的区间

#### 解法

​		排序，遍历数组，对满足的情况进行合并，不满足合并的点加入答案		

```c++
class Solution {
public:
    static bool cmp(const vector<int>& a, const vector<int>& b){
        if(a[0] != b[0])
            return a[0] < b[0];
        return a[1] < b[1];
    }
    
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int> > ans;
        if(intervals.size() == 0)
            return ans;
        sort(intervals.begin(), intervals.end(), cmp);
        vector<int> cur = intervals[0];
        for(int i=1; i<intervals.size(); i++){
            if(intervals[i][0] > cur[1]){
                ans.push_back(cur);
                cur = intervals[i];
            }
            else if(intervals[i][1] > cur[1])
                cur[1] = intervals[i][1];
        }
        ans.push_back(cur);
        return ans;
    }
};
```



## 一些hard

### 41. First Missing Positive

#### 题目描述

​		给一个数组，返回第一个没出现的正数

#### 解法

​		把在数组长度范围之内的正数放到它应该在的位置上，然后遍历一遍，去看哪个正数没有出现

```c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        for(int i=0; i<nums.size(); i++){
            while(nums[i] >=1 && nums[i] <=nums.size() && nums[nums[i] - 1] != nums[i]){
                int tmp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = tmp;
            }
        }
        for(int i=0; i<nums.size(); i++)
            if(nums[i] != i+1)
                return i + 1;
        return nums.size() + 1;
    }
};
```

### 49. Group Anagrams

### 题目描述

​		将组成字母相同的字符串放到一个vector里面

#### 解法

​		哈希

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string> > hash;
        for(int i=0; i<strs.size(); i++){
            string tmp = strs[i];
            sort(tmp.begin(), tmp.end());
            hash[tmp].push_back(strs[i]);
        }
        vector<vector<string> > ans;
        for(auto it=hash.begin(); it!=hash.end(); it++){
            ans.push_back(it->second);
        }
        return ans;
    }
};
```



## 双指针

### 15. 3Sum

#### 题目描述

​		给一个数组，求和为0的三元组，要求不重复。

####　解法

​		枚举第一个数，然后对剩下两个值进行双指针处理，注意去重。

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int> > ans;
        int len = nums.size();
        int l, r;
        for(int i=0; i<len-2; i++){
            cout << i;
            cout << "???";
            if(i!=0 && nums[i] == nums[i-1])
                continue;
            l = i + 1, r = nums.size() - 1;
            while(l < r){
                int tmp =nums[i] + nums[l] + nums[r];
                if(tmp == 0){
                    vector<int>tmp{nums[i], nums[l], nums[r]};
                    ans.push_back(tmp);
                    while(l<r && nums[l+1] == nums[l])
                        l++;
                    l++;
                }
                else if(tmp > 0){
                    while(r>l && nums[r-1] == nums[r])
                        r--;
                    r--;
                }
                else{
                    while(l<r && nums[l+1] == nums[l])
                        l++;
                    l++;
                }
            }
        }
        return ans;
    }
};
```



## 暴力枚举

### * 5. Longest Palindromic Substring

#### 题目描述

​		输入一个字符串s，返回其中的最长子字符串。

#### 解法

​		暴力枚举，分两种情况，一种为这个字符串为奇数长度，一种为偶数长度。这两种情况导致了中心是一个还是两个字符，有所区别。

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        int res = 0;
        string ret;
        for(int i=0; i<s.size(); i++){
            for(int j=0; i - j >= 0 && i + j < s.size(); j++){
                if(s[i-j] == s[i+j]){
                    if(2 * j + 1 > res){
                        res = 2 * j + 1;
                        ret = s.substr(i - j, res);
                    }
                }
                else break;
            }
            for(int j=i, k=i+1; j>=0 && k<s.size(); j--, k++){
                if(s[j] == s[k]){
                    if(k-j+1>res){
                        res = k - j + 1;
                        ret = s.substr(j, res);
                    }
                }
                else  break;
            }
        }
        return ret;
    }
};
```

## 无聊的题目

### 38. Count and Say

#### 题目描述

​		查上一个数字符串中连续出现的数字个数，拼成字符串

#### 解法

​		很无聊，自己看吧

```c++
class Solution {
public:
    string countAndSay(int n) {
        if(n==1)
            return "1";
        string s = "1*", ret;
        int count;
        for(int i=2; i<=n; i++){
            ret = "";
            count = 1;    
            for(int j=0; j<s.size() - 1; j++){
                if(s[j] == s[j+1])
                    count ++;
                else{
                    ret += to_string(count) + s[j];
                    count = 1;
                }
            }
            s = ret + '*';
        }
        return ret;
    }
};
```

