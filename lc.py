1. Two Sum
Easy

23045

774

Add to List

Share
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.



Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]


Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.



class Solution:
    def twoSum(self, nums, target):
        result = []
        record_dict = {}

        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in record_dict:
                return [record_dict[diff], i]
            record_dict[nums[i]] = i


##################################################################################################################################

20. Valid Parentheses
Easy

8385

340

Add to List

Share
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.


Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
Example 4:

Input: s = "([)]"
Output: false
Example 5:

Input: s = "{[]}"
Output: true


Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.



class Solution:
    def isValid(self, s: str) -> bool:
        len_s = len(s)
        stack = []

        for i in range(len(s)):
            if s[i] == '(' or s[i] == '[' or s[i] == '{':
                stack.append(s[i])
            else:
                if not stack:
                    return False
                else:
                    if stack[-1] == '(' and s[i] != ')':
                        return False
                    if stack[-1] == '[' and s[i] != ']':
                        return False
                    if stack[-1] == '{' and s[i] != '}':
                        return False
                    stack.pop()

        if not stack:
            return True
        return False




##################################################################################################################################

32. Longest Valid Parentheses
Hard

5735

198

Add to List

Share
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.



Example 1:

Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".
Example 2:

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
Example 3:

Input: s = ""
Output: 0


Constraints:

0 <= s.length <= 3 * 104
s[i] is '(', or ')'.



class Solution:
    def longestValidParentheses(self, s: str) -> int:

        stack = list(s)
        len_stack = len(stack)
        count = 0
        max_count = 0

        for i in range(len_stack-1):
            if stack[i] == '(' and stack[i+1] == ')':
                count += 2
                max_count = max(count, max_count)
            elif stack[i] == '(' and stack[i+1] == '(':
                count = 0
            elif stack[i] == ')' and stack[i+1] == ')':
                count = 0
        return max_count



##################################################################################################################################

46. Permutations
Medium

6943

141

Add to List

Share
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.



Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]


Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.



class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []

        if len(nums) == 1:
            return [nums]

        result = []
        for i in range(len(nums)):
            curr = nums[i]
            remain_nums = nums[:i] + nums[i+1:]
            for p in self.permute(remain_nums):
                result.append([curr]+p)
        return result


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        def dfs(res, nums, path):
            if len(nums)<=1:
                res.append(path+nums)
                return
            for i, v in enumerate(nums):
                dfs(res,nums[:i]+nums[i+1:],path+[v])

        res = []
        path = []
        dfs(res, nums, path)
        return res


##################################################################################################################################
74. Search a 2D Matrix
Medium

3921

212

Add to List

Share
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.


Example 1:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
Example 2:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false


Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 100
-104 <= matrix[i][j], target <= 104




class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row = 0
        result = False

        if len(matrix) == 1:
            return target in matrix[0]

        for i in range(len(matrix)-1):
            if target >= matrix[i][0] and target <= matrix[i+1][0]:
                if target in matrix[i]:
                    result = True
                else:
                    result = False
        if not result:
            return target in matrix[-1]
        return result

##################################################################################################################################

102. Binary Tree Level Order Traversal
Medium

5378

115

Add to List

Share
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).



Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
Example 2:

Input: root = [1]
Output: [[1]]
Example 3:

Input: root = []
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 2000].
-1000 <= Node.val <= 1000



class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        def bfs(node, level):
            if len(result) == level:
                result.append([])

            result[level].append(node.val)

            if node.left:
                bfs(node.left, level + 1)
            if node.right:
                bfs(node.right, level + 1)

        if not root:
            return []

        result = []
        bfs(root, 0)
        return result


##################################################################################################################################
21. Merge Two Sorted Lists
Easy

7713

828

Add to List

Share
Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.



Example 1:


Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: l1 = [], l2 = []
Output: []
Example 3:

Input: l1 = [], l2 = [0]
Output: [0]


Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both l1 and l2 are sorted in non-decreasing order.



class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        curr = dummy = ListNode()

        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next

        if not l1:
            curr.next = l2
        else:
            curr.next = l1

        return dummy.next

##################################################################################################################################

129. Sum Root to Leaf Numbers
Medium

2630

64

Add to List

Share
You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.



Example 1:


Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
Example 2:


Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.


Constraints:

The number of nodes in the tree is in the range [1, 1000].
0 <= Node.val <= 9
The depth of the tree will not exceed 10.


class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def keepLookingTotalSum(root, total_sum):
            if not root:
                return 0
            total_sum = total_sum * 10 + root.val
            if not root.left and not root.right:
                return total_sum
            else:
                return keepLookingTotalSum(root.left, total_sum) + keepLookingTotalSum(root.right, total_sum)

        total_sum = 0
        total_sum = keepLookingTotalSum(root, total_sum)
        return total_sum


#################################################################################################################################


547. Number of Provinces
Medium

3471

196

Add to List

Share
There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.



Example 1:


Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2
Example 2:


Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3


Constraints:

1 <= n <= 200
n == isConnected.length
n == isConnected[i].length
isConnected[i][j] is 1 or 0.
isConnected[i][i] == 1
isConnected[i][j] == isConnected[j][i]




class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
#         [[1,1,0],
#          [1,1,0],
#          [0,0,1]]

#         [[1,0,0],
#          [0,1,0],
#          [0,0,1]]
        count = 0
        for r, row in enumerate(isConnected):
            for c, col in enumerate(row):
                if isConnected[r][c] == 1:
                    self.keepchecking(r, c, isConnected)
                    count += 1
        return count

    def keepchecking(self, r, c, isConnected):
        isConnected[r][c] = 0
        if (r+1<len(isConnected)) and isConnected[r+1][c] == 1:
            self.keepchecking(r+1, c, isConnected)
        if (c+1<len(isConnected[0])) and isConnected[r][c+1] == 1:
            self.keepchecking(r, c+1, isConnected)
        if (r-1>=0) and isConnected[r-1][c] == 1:
            self.keepchecking(r-1, c, isConnected)
        if (c-1>=0) and isConnected[r][c-1] == 1:
            self.keepchecking(r, c-1, isConnected)

#################################################################################################################################


1041. Robot Bounded In Circle
Medium

1574

409

Add to List

Share
On an infinite plane, a robot initially stands at (0, 0) and faces north. The robot can receive one of three instructions:

"G": go straight 1 unit;
"L": turn 90 degrees to the left;
"R": turn 90 degrees to the right.
The robot performs the instructions given in order, and repeats them forever.

Return true if and only if there exists a circle in the plane such that the robot never leaves the circle.



Example 1:

Input: instructions = "GGLLGG"
Output: true
Explanation: The robot moves from (0,0) to (0,2), turns 180 degrees, and then returns to (0,0).
When repeating these instructions, the robot remains in the circle of radius 2 centered at the origin.
Example 2:

Input: instructions = "GG"
Output: false
Explanation: The robot moves north indefinitely.
Example 3:

Input: instructions = "GL"
Output: true
Explanation: The robot moves from (0, 0) -> (0, 1) -> (-1, 1) -> (-1, 0) -> (0, 0) -> ...


Constraints:

1 <= instructions.length <= 100
instructions[i] is 'G', 'L' or, 'R'.



class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        # directions: 0-North, 1-East, 2-South, 3-West
        x, y, direction = 0, 0, 0
        for i in instructions:
            if i == "L":
                direction = (direction - 1) % 4
            elif i == "R":
                direction = (direction + 1) % 4
            elif i == "G":
                if direction == 0:
                    y += 1
                elif direction == 1:
                    x += 1
                elif direction == 2:
                    y -= 1
                elif direction == 3:
                    x -= 1
        return (x==0 and y==0) or direction != 0


#################################################################################################################################
202. Happy Number
Easy

3551

559

Add to List

Share
Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.



Example 1:

Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
Example 2:

Input: n = 2
Output: false


Constraints:

1 <= n <= 231 - 1


class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum(int(i) ** 2 for i in str(n))
        return n == 1

#################################################################################################################################

1120. Maximum Average Subtree
Medium

487

15

Add to List

Share
Given the root of a binary tree, find the maximum average value of any subtree of that tree.

(A subtree of a tree is any node of that tree plus all its descendants. The average value of a tree is the sum of its values, divided by the number of nodes.)



Example 1:



Input: [5,6,1]
Output: 6.00000
Explanation:
For the node with value = 5 we have an average of (5 + 6 + 1) / 3 = 4.
For the node with value = 6 we have an average of 6 / 1 = 6.
For the node with value = 1 we have an average of 1 / 1 = 1.
So the answer is 6 which is the maximum.


Note:

The number of nodes in the tree is between 1 and 5000.
Each node will have a value between 0 and 100000.
Answers will be accepted as correct if they are within 10^-5 of the correct answer.



class Solution:
    def maximumAverageSubtree(self, root: TreeNode) -> float:
        def dfs(root):
            if not root:
                return 0, 0

            left_sum, left_count = dfs(root.left)
            right_sum, right_count = dfs(root.right)
            curr_sum = root.val + left_sum + right_sum
            curr_count = 1 + left_count + right_count

            self.res = max(self.res, curr_sum/curr_count)
            return curr_sum, curr_count


        if not root:
            return 0

        self.res = 0
        dfs(root)
        return self.res


#################################################################################################################################
543. Diameter of Binary Tree
Easy

5338

326

Add to List

Share
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.



Example 1:


Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
Example 2:

Input: root = [1,2]
Output: 1


Constraints:

The number of nodes in the tree is in the range [1, 104].
-100 <= Node.val <= 100




class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def dfs(root):
            nonlocal diameter
            if not root:
                return 0

            left = dfs(root.left)
            right = dfs(root.right)

            diameter = max(diameter, left+right)
            return max(left, right)+1

        diameter = 0
        dfs(root)
        return diameter

#################################################################################################################################

240. Search a 2D Matrix II
Medium

5285

97

Add to List

Share
Write an efficient algorithm that searches for a target value in an m x n integer matrix. The matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.


Example 1:


Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true
Example 2:


Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
Output: false


Constraints:

m == matrix.length
n == matrix[i].length
1 <= n, m <= 300
-109 <= matix[i][j] <= 109
All the integers in each row are sorted in ascending order.
All the integers in each column are sorted in ascending order.
-109 <= target <= 109

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        height = len(matrix)
        width = len(matrix[0])
        if not height or not width:
            return False

        row = height - 1
        col = 0

        while col < width and row >= 0:
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else: # found it
                return True
        return False


#################################################################################################################################

215. Kth Largest Element in an Array
Medium

6363

384

Add to List

Share
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.



Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4


Constraints:

1 <= k <= nums.length <= 104
-104 <= nums[i] <= 104



#fatboy
# Approach #1
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-k]

# Approach #2
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]


#################################################################################################################################
347. Top K Frequent Elements
Medium

5649

284

Add to List

Share
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.



Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]


Constraints:

1 <= nums.length <= 105
k is in the range [1, the number of unique elements in the array].
It is guaranteed that the answer is unique.


class Solution:
    import heapq
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        heap =[]
        record = {}
        for num in nums:
            if num in record:
                record[num] += 1
            else:
                record[num] = 1
        for key,val in record.items():
            heapq.heappush(heap,(val, key))
            if len(heap) > k:
                heapq.heappop(heap)
        return [el[1] for el in heap]

#################################################################################################################################
111. Minimum Depth of Binary Tree
Easy

2789

847

Add to List

Share
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.



Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: 2
Example 2:

Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5


Constraints:

The number of nodes in the tree is in the range [0, 105].
-1000 <= Node.val <= 1000


class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        left = self.minDepth(root.left)
        right = self.minDepth(root.right)

        if left and right:
            return 1 + min(left,right)
        elif not left and right:
            return 1 + right
        elif not right and left:
            return 1 + left
        else:
            return 1


#################################################################################################################################
1448. Count Good Nodes in Binary Tree
Medium

1177

45

Add to List

Share
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.



Example 1:



Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.
Example 2:



Input: root = [3,3,null,4,2]
Output: 3
Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.
Example 3:

Input: root = [1]
Output: 1
Explanation: Root is considered as good.


Constraints:

The number of nodes in the binary tree is in the range [1, 10^5].
Each node's value is between [-10^4, 10^4].


class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, max_val):
            nonlocal count
            if not node:
                return
            if node.val >= max_val:
                max_val = node.val
                count += 1
            dfs(node.left, max_val)
            dfs(node.right, max_val)

        count = 0
        dfs(root, float('-inf'))
        return count

#################################################################################################################################
695. Max Area of Island
Medium

3890

115

Add to List

Share
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.



Example 1:


Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
Example 2:

Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0


Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 50
grid[i][j] is either 0 or 1.




#fatboy cheated
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def keepLooking(r, c, grid):
            if (r < 0 or r >= len(grid)) or (c < 0 or c >= len(grid[0])):
                return 0
            if grid[r][c] == 0:
                return 0
            grid[r][c] = 0

            return keepLooking(r+1, c, grid) + keepLooking(r, c+1, grid) + keepLooking(r-1, c, grid) + keepLooking(r, c-1, grid) + 1

        max_area = 0

        for r, row in enumerate(grid):
            for c, col in enumerate(row):
                if grid[r][c] == 1:
                    max_area = max(max_area, keepLooking(r, c, grid))
        return max_area


#################################################################################################################################
567. Permutation in String
Medium

2752

83

Add to List

Share
Given two strings s1 and s2, return true if s2 contains the permutation of s1.

In other words, one of s1's permutations is the substring of s2.



Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false


Constraints:

1 <= s1.length, s2.length <= 104
s1 and s2 consist of lowercase English letters.



class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        def dfs(s1):
            result = []
            if not s1:
                return ''

            if len(s1) == 1:
                return s1

            for i in range(len(s1)):
                curr = s1[i]
                remain = s1[:i] + s1[i+1:]
                for r in dfs(remain):
                    result.append(curr+r)
            return result


        result = dfs(s1)
        for s in result:
            if s in s2:
                return True
        return False

#################################################################################################################################
125. Valid Palindrome
Easy

2280

4132

Add to List

Share
Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.



Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.


Constraints:

1 <= s.length <= 2 * 105
s consists only of printable ASCII characters.



class Solution:
    def isPalindrome(self, s: str) -> bool:
        filtered_str_lst = filter(str.isalnum, s)
        clean_s = ''.join(filtered_str_lst)
        clean_lower_s = clean_s.lower()
        return clean_lower_s == clean_lower_s[::-1]

#################################################################################################################################
146. LRU Cache
Medium

9415

372

Add to List

Share
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.



Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4


Constraints:

1 <= capacity <= 3000
0 <= key <= 104
0 <= value <= 105
At most 2 * 105 calls will be made to get and put.



from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity


    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

#################################################################################################################################
121. Best Time to Buy and Sell Stock
Easy

9804

400

Add to List

Share
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.



Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.


Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104
Accepted


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_buy = prices[0]
        max_return = 0
        for price in prices:
            min_buy = min(min_buy, price)
            curr_return = price - min_buy
            max_return = max(curr_return, max_return)
        return max_return


#################################################################################################################################
122. Best Time to Buy and Sell Stock II
Easy

4907

2119

Add to List

Share
You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).



Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e., max profit = 0.


Constraints:

1 <= prices.length <= 3 * 104
0 <= prices[i] <= 104


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max = 0
        for i in range(len(prices)-1):
            if prices[i+1] - prices[i] > 0:
                max += prices[i+1] - prices[i]
        return max

#################################################################################################################################
124. Binary Tree Maximum Path Sum
Hard

6541

426

Add to List

Share
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any path.



Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.


Constraints:

The number of nodes in the tree is in the range [1, 3 * 104].
-1000 <= Node.val <= 1000


class Solution:
    def maxPathSum(self, root):
        def keepLookingMaxSum(root):
            nonlocal max_sum
            if not root:
                return 0

            left_sum = max(keepLookingMaxSum(root.left), 0)
            right_sum = max(keepLookingMaxSum(root.right), 0)
            curr_max_sum = root.val + left_sum + right_sum
            max_sum = max(max_sum, curr_max_sum)
            return root.val + max(left_sum, right_sum)

        max_sum = float('-inf')
        keepLookingMaxSum(root)
        return max_sum

#################################################################################################################################
101. Symmetric Tree
Easy

6869

179

Add to List

Share
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).



Example 1:


Input: root = [1,2,2,3,4,4,3]
Output: true
Example 2:


Input: root = [1,2,2,null,3,null,3]
Output: false


Constraints:

The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100

class Solution:
    def isSymmetric(self, root):
        def keepCheckIsSymmetric(left,right):
            if not left and not right:
                return True
            if left and right and left.val == right.val:
                return keepCheckIsSymmetric(left.left, right.right) and keepCheckIsSymmetric(left.right, right.left)
            return False
        return keepCheckIsSymmetric(root, root)

#################################################################################################################################

200. Number of Islands
Medium

9501

260

Add to List

Share
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.



Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3


Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.




class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for r, row in enumerate(grid):
            for c, col in enumerate(row):
                if grid[r][c] == '1':
                    self.removeNeighbors(r,c,grid)
                    count += 1
        return count

    def removeNeighbors(self, r, c, grid):
        grid[r][c] = '0'
        if r+1<len(grid) and grid[r+1][c] == '1':
            self.removeNeighbors(r+1,c,grid)
        if r-1>=0 and grid[r-1][c] == '1':
            self.removeNeighbors(r-1,c,grid)
        if c+1<len(grid[0]) and grid[r][c+1] == '1':
            self.removeNeighbors(r,c+1,grid)
        if c-1>=0 and grid[r][c-1] == '1':
            self.removeNeighbors(r,c-1,grid)


#################################################################################################################################

56. Merge Intervals
Medium

8565

409

Add to List

Share
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.



Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.


Constraints:

1 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 104


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])

        result = []
        for interval in intervals:
            if not result or result[-1][1] < interval[0]:
                result.append(interval)
            else:
                result[-1][1] = max(result[-1][1], interval[1])
        return result

#################################################################################################################################
2. Add Two Numbers
Medium

13095

2958

Add to List

Share
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.



Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]


Constraints:

The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading zeros.


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        curr = result = ListNode()

        while l1 or l2 or carry:
            digit = 0
            if l1:
                digit += l1.val
                l1 = l1.next
            if l2:
                digit += l2.val
                l2 = l2.next
            if carry:
                digit += carry

            carry = digit // 10
            digit = digit % 10
            curr.next = ListNode(digit)
            curr = curr.next

        return result.next

########################################################################################

3. Longest Substring Without Repeating Characters
Medium

16234

789

Add to List

Share
Given a string s, find the length of the longest substring without repeating characters.



Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
Example 4:

Input: s = ""
Output: 0


Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        reference = []

        for letter in s:
            if letter not in reference:
                reference.append(letter)
            else:
                prev_index = reference.index(letter)
                reference = reference[prev_index+1:] + [letter]
            max_len = max(len(reference), max_len)
        return max_len


########################################################################################

560. Subarray Sum Equals K
Medium

8411

286

Add to List

Share
Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.



Example 1:

Input: nums = [1,1,1], k = 2
Output: 2
Example 2:

Input: nums = [1,2,3], k = 3
Output: 2


Constraints:

1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107


from collections import defaultdict
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        record_sum = defaultdict(int)
        count = 0
        curr_sum = 0

        for num in nums:
            curr_sum += num
            count += record_sum[curr_sum - k]
            record_sum[curr_sum] += 1
        count += record_sum[k]
        return count

########################################################################################
253. Meeting Rooms II
Medium

4153

74

Add to List

Share
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.



Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1


Constraints:

1 <= intervals.length <= 104
0 <= starti < endi <= 106



class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x:x[0])
        conf_rooms = []
        heapq.heappush(conf_rooms, intervals[0][1])
        for interval in intervals[1:]:
            if conf_rooms[0] <= interval[0]:
                heapq.heappop(conf_rooms)
            heapq.heappush(conf_rooms, interval[1])
        return len(conf_rooms)

########################################################################################

733. Flood Fill
Easy

2300

265

Add to List

Share
An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.

You are also given three integers sr, sc, and newColor. You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with newColor.

Return the modified image after performing the flood fill.



Example 1:


Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.
Example 2:

Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, newColor = 2
Output: [[2,2,2],[2,2,2]]


Constraints:

m == image.length
n == image[i].length
1 <= m, n <= 50
0 <= image[i][j], newColor < 216
0 <= sr < m
0 <= sc < n


class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        def keepLookingOldColor(r, c):
            if image[r][c] == oldColor:
                image[r][c] = newColor
                if r+1 < len(image):
                    keepLookingOldColor(r+1, c)
                if c+1 < len(image[0]):
                    keepLookingOldColor(r, c+1)
                if r-1 >= 0:
                    keepLookingOldColor(r-1, c)
                if c-1 >= 0:
                    keepLookingOldColor(r, c-1)

        oldColor = image[sr][sc]
        if newColor == oldColor:
            return image
        keepLookingOldColor(sr, sc)
        return image


########################################################################################

973. K Closest Points to Origin
Medium

3511

169

Add to List

Share
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).



Example 1:


Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.


Constraints:

1 <= k <= points.length <= 104
-104 < xi, yi < 104


import math
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key = lambda x: math.sqrt(x[0] ** 2 + x[1] ** 2))
        return points[0:k]

########################################################################################
545. Boundary of Binary Tree
Medium

844

1404

Add to List

Share
The boundary of a binary tree is the concatenation of the root, the left boundary, the leaves ordered from left-to-right, and the reverse order of the right boundary.

The left boundary is the set of nodes defined by the following:

The root node's left child is in the left boundary. If the root does not have a left child, then the left boundary is empty.
If a node in the left boundary and has a left child, then the left child is in the left boundary.
If a node is in the left boundary, has no left child, but has a right child, then the right child is in the left boundary.
The leftmost leaf is not in the left boundary.
The right boundary is similar to the left boundary, except it is the right side of the root's right subtree. Again, the leaf is not part of the right boundary, and the right boundary is empty if the root does not have a right child.

The leaves are nodes that do not have any children. For this problem, the root is not a leaf.

Given the root of a binary tree, return the values of its boundary.



Example 1:


Input: root = [1,null,2,3,4]
Output: [1,3,4,2]
Explanation:
- The left boundary is empty because the root does not have a left child.
- The right boundary follows the path starting from the root's right child 2 -> 4.
  4 is a leaf, so the right boundary is [2].
- The leaves from left to right are [3,4].
Concatenating everything results in [1] + [] + [3,4] + [2] = [1,3,4,2].
Example 2:


Input: root = [1,2,3,4,5,6,null,null,null,7,8,9,10]
Output: [1,2,4,7,8,9,10,6,3]
Explanation:
- The left boundary follows the path starting from the root's left child 2 -> 4.
  4 is a leaf, so the left boundary is [2].
- The right boundary follows the path starting from the root's right child 3 -> 6 -> 10.
  10 is a leaf, so the right boundary is [3,6], and in reverse order is [6,3].
- The leaves from left to right are [4,7,8,9,10].
Concatenating everything results in [1] + [2] + [4,7,8,9,10] + [6,3] = [1,2,4,7,8,9,10,6,3].


Constraints:

The number of nodes in the tree is in the range [1, 104].
-1000 <= Node.val <= 1000


class Solution:
    def __init__(self):
        self.result = []

    def boundaryLeft(self,node):
        if not node:
            return
        if node.left:
            self.result.append(node.val)
            self.boundaryLeft(node.left)
        elif node.right:
            self.result.append(node.val)
            self.boundaryLeft(node.right)

    def boundaryLeaves(self,node):
        if not node:
            return
        self.boundaryLeaves(node.left)
        if not node.left and not node.right:
            self.result.append(node.val)
        self.boundaryLeaves(node.right)

    def boundaryRight(self,node):
        if not node:
            return
        if node.right:
            self.boundaryRight(node.right)
            self.result.append(node.val)
        elif node.left:
            self.boundaryRight(node.left)
            self.result.append(node.val)


    def boundaryOfBinaryTree(self, root: TreeNode) -> List[int]:
        if root:
            self.result.append(root.val)
            self.boundaryLeft(root.left)
            self.boundaryLeaves(root.left)
            self.boundaryLeaves(root.right)
            self.boundaryRight(root.right)
        return self.result

########################################################################################
849. Maximize Distance to Closest Person
Medium

1489

137

Add to List

Share
You are given an array representing a row of seats where seats[i] = 1 represents a person sitting in the ith seat, and seats[i] = 0 represents that the ith seat is empty (0-indexed).

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized.

Return that maximum distance to the closest person.



Example 1:


Input: seats = [1,0,0,0,1,0,1]
Output: 2
Explanation:
If Alex sits in the second open seat (i.e. seats[2]), then the closest person has distance 2.
If Alex sits in any other open seat, the closest person has distance 1.
Thus, the maximum distance to the closest person is 2.
Example 2:

Input: seats = [1,0,0,0]
Output: 3
Explanation:
If Alex sits in the last seat (i.e. seats[3]), the closest person is 3 seats away.
This is the maximum distance possible, so the answer is 3.
Example 3:

Input: seats = [0,1]
Output: 1


Constraints:

2 <= seats.length <= 2 * 104
seats[i] is 0 or 1.
At least one seat is empty.
At least one seat is occupied.


class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        exists = [i for i in range(len(seats)) if seats[i]]
        diffs = [exists[i+1]-exists[i] for i in range(len(exists)-1)] if len(exists) > 1 else [0]
        return max(max(diffs)//2, exists[0], len(seats)-1-exists[-1])

########################################################################################

937. Reorder Data in Log Files
Easy

1236

3163

Add to List

Share
You are given an array of logs. Each log is a space-delimited string of words, where the first word is the identifier.

There are two types of logs:

Letter-logs: All words (except the identifier) consist of lowercase English letters.
Digit-logs: All words (except the identifier) consist of digits.
Reorder these logs so that:

The letter-logs come before all digit-logs.
The letter-logs are sorted lexicographically by their contents. If their contents are the same, then sort them lexicographically by their identifiers.
The digit-logs maintain their relative ordering.
Return the final order of the logs.



Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
Explanation:
The letter-log contents are all different, so their ordering is "art can", "art zero", "own kit dig".
The digit-logs have a relative order of "dig1 8 1 5 1", "dig2 3 6".
Example 2:

Input: logs = ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]


Constraints:

1 <= logs.length <= 100
3 <= logs[i].length <= 100
All the tokens of logs[i] are separated by a single space.
logs[i] is guaranteed to have an identifier and at least one word after the identifier.


class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters=[]
        digits=[]
        for log in logs:
            if log[-1].isdigit():
                digits.append(log)
            else:
                letters.append(log)
        letters = sorted(letters, key=lambda letter: (letter.split()[1:],letter.split()[0]))
        return letters+digits

########################################################################################
1228. Missing Number In Arithmetic Progression
Easy

204

25

Add to List

Share
In some array arr, the values were in arithmetic progression: the values arr[i + 1] - arr[i] are all equal for every 0 <= i < arr.length - 1.

A value from arr was removed that was not the first or last value in the array.

Given arr, return the removed value.



Example 1:

Input: arr = [5,7,11,13]
Output: 9
Explanation: The previous array was [5,7,9,11,13].
Example 2:

Input: arr = [15,13,12]
Output: 14
Explanation: The previous array was [15,14,13,12].


Constraints:

3 <= arr.length <= 1000
0 <= arr[i] <= 105
The given array is guaranteed to be a valid array.


class Solution:
    def missingNumber(self, arr: List[int]) -> int:
        len_arr = len(arr)
        total_diff = arr[-1] - arr[0]
        avg_diff = total_diff // len_arr

        for i in range(len_arr-1):
            if arr[i+1] - arr[i] != avg_diff:
                return arr[i] + avg_diff
        return arr[i] + avg_diff

########################################################################################

1200. Minimum Absolute Difference
Easy

729

35

Add to List

Share
Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements.

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows

a, b are from arr
a < b
b - a equals to the minimum absolute difference of any two elements in arr


Example 1:

Input: arr = [4,2,1,3]
Output: [[1,2],[2,3],[3,4]]
Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
Example 2:

Input: arr = [1,3,6,10,15]
Output: [[1,3]]
Example 3:

Input: arr = [3,8,-10,23,19,-4,-14,27]
Output: [[-14,-10],[19,23],[23,27]]


Constraints:

2 <= arr.length <= 10^5
-10^6 <= arr[i] <= 10^6


class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        result = []
        min_diff = float('inf')
        len_arr = len(arr)
        arr.sort()

        for i in range(len_arr-1):
            curr_diff = arr[i+1] - arr[i]
            if curr_diff < min_diff:
                result = [[arr[i], arr[i+1]]]
                min_diff = curr_diff
            elif curr_diff == min_diff:
                result.append([arr[i], arr[i+1]])
        return result

########################################################################################
