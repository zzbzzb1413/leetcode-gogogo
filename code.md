## 二分

### 69.sqrt(x)

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



### 33. Search in Rotated Sorted Array

#### 题目描述		

​		在一个旋转数组中找一个目标值

#### 解法

​		









