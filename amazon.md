# LeetCode Solutions with Explanations

## 1603. Design Parking System (Easy)

**Problem:** [LeetCode 1603 - Design Parking System](https://leetcode.com/problems/design-parking-system/)

### Solution with Explanation

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        # Store remaining slots by car type (1-based index)
        self.slots = [0, big, medium, small]

    def addCar(self, carType: int) -> bool:
        # If there’s space, park the car and decrement the slot
        if self.slots[carType] > 0:
            self.slots[carType] -= 1
            return True
        return False
```

---

## 1710. Maximum Units on a Truck (Easy)

**Problem:** [LeetCode 1710 - Maximum Units on a Truck](https://leetcode.com/problems/maximum-units-on-a-truck/)

### Solution with Explanation

```python
def maximumUnits(boxTypes: List[List[int]], truckSize: int) -> int:
    # Sort box types by units per box in descending order
    boxTypes.sort(key=lambda x: -x[1])
    units = 0

    for boxes, units_per_box in boxTypes:
        if truckSize == 0:
            break
        # Load as many boxes as possible
        count = min(boxes, truckSize)
        units += count * units_per_box
        truckSize -= count

    return units
```

---

## 2357. Make Array Zero by Subtracting Equal Amounts (Easy)

**Problem:** [LeetCode 2357 - Make Array Zero by Subtracting Equal Amounts](https://leetcode.com/problems/make-array-zero-by-subtracting-equal-amounts/)

### Solution with Explanation

```python
def minimumOperations(nums: List[int]) -> int:
    # Count how many unique non-zero numbers there are
    return len(set(num for num in nums if num != 0))
```

---

## 49. Group Anagrams (Medium)

**Problem:** [LeetCode 49 - Group Anagrams](https://leetcode.com/problems/group-anagrams/)

### Solution with Explanation

```python
from collections import defaultdict

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    # Dictionary to group words by their sorted character tuple
    anagram_map = defaultdict(list)

    for word in strs:
        key = tuple(sorted(word))  # Sorted characters as key
        anagram_map[key].append(word)

    return list(anagram_map.values())
```

---

## 138. Copy List with Random Pointer (Medium)

**Problem:** [LeetCode 138 - Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

### Solution with Explanation

```python
class Node:
    def __init__(self, val: int, next: 'Node' = None, random: 'Node' = None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head: 'Optional[Node]') -> 'Optional[Node]':
    if not head:
        return None

    # Step 1: Clone and interleave nodes
    current = head
    while current:
        cloned = Node(current.val, current.next)
        current.next = cloned
        current = cloned.next

    # Step 2: Set random pointers for cloned nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next

    # Step 3: Detach cloned list
    original = head
    clone = head.next
    clone_head = clone

    while original:
        original.next = original.next.next
        if clone.next:
            clone.next = clone.next.next
        original = original.next
        clone = clone.next

    return clone_head
```
# LeetCode Solutions with Explanations (Continued)

## 146. LRU Cache (Medium)

**Problem:** [LeetCode 146 - LRU Cache](https://leetcode.com/problems/lru-cache/)

### Solution with Explanation

```python
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move key to end to mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        self.cache[key] = value
        # Remove least recently used if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

## 200. Number of Islands (Medium)

**Problem:** [LeetCode 200 - Number of Islands](https://leetcode.com/problems/number-of-islands/)

### Solution with Explanation

```python
def numIslands(grid: List[List[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        # Mark as visited
        grid[r][c] = '0'
        # Visit all neighbors
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1

    return count
```

---

## 207. Course Schedule (Medium)

**Problem:** [LeetCode 207 - Course Schedule](https://leetcode.com/problems/course-schedule/)

### Solution with Explanation

```python
from collections import defaultdict, deque

def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = defaultdict(list)
    indegree = [0] * numCourses

    # Build graph and count indegrees
    for dest, src in prerequisites:
        graph[src].append(dest)
        indegree[dest] += 1

    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1
        for neighbor in graph[course]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return completed == numCourses
```

---

## 253. Meeting Rooms II (Medium)

**Problem:** [LeetCode 253 - Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

### Solution with Explanation

```python
import heapq

def minMeetingRooms(intervals: List[List[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])
    heap = []

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)  # Reuse room
        heapq.heappush(heap, end)  # Allocate new room or extend existing

    return len(heap)
```

---

## 348. Design Tic-Tac-Toe (Medium)

**Problem:** [LeetCode 348 - Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)

### Solution with Explanation

```python
class TicTacToe:

    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.anti_diag = 0

    def move(self, row: int, col: int, player: int) -> int:
        to_add = 1 if player == 1 else -1

        self.rows[row] += to_add
        self.cols[col] += to_add

        if row == col:
            self.diag += to_add
        if row + col == self.n - 1:
            self.anti_diag += to_add

        if (abs(self.rows[row]) == self.n or
            abs(self.cols[col]) == self.n or
            abs(self.diag) == self.n or
            abs(self.anti_diag) == self.n):
            return player

        return 0
```
# LeetCode Solutions with Explanations (Continued)

## 545. Boundary of Binary Tree (Medium)

**Problem:** [LeetCode 545 - Boundary of Binary Tree](https://leetcode.com/problems/boundary-of-binary-tree/)

### Solution with Explanation

```python
from typing import List

def boundaryOfBinaryTree(root: 'TreeNode') -> List[int]:
    if not root:
        return []

    res = [root.val]

    def leftBoundary(node):
        while node:
            if node.left or node.right:
                res.append(node.val)
            if node.left:
                node = node.left
            else:
                node = node.right

    def leaves(node):
        if not node:
            return
        if not node.left and not node.right:
            res.append(node.val)
        leaves(node.left)
        leaves(node.right)

    def rightBoundary(node):
        stack = []
        while node:
            if node.left or node.right:
                stack.append(node.val)
            if node.right:
                node = node.right
            else:
                node = node.left
        while stack:
            res.append(stack.pop())

    if root.left:
        leftBoundary(root.left)
    leaves(root.left)
    leaves(root.right)
    if root.right:
        rightBoundary(root.right)

    return res
```

---

## 767. Reorganize String (Medium)

**Problem:** [LeetCode 767 - Reorganize String](https://leetcode.com/problems/reorganize-string/)

### Solution with Explanation

```python
import heapq
from collections import Counter

def reorganizeString(s: str) -> str:
    count = Counter(s)
    max_heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(max_heap)

    prev_freq, prev_char = 0, ''
    result = []

    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)

        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))

        freq += 1  # decrement frequency since freq is negative
        prev_freq, prev_char = freq, char

    result_str = ''.join(result)
    if len(result_str) != len(s):
        return ""
    return result_str
```

---

## 863. All Nodes Distance K in Binary Tree (Medium)

**Problem:** [LeetCode 863 - All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

### Solution with Explanation

```python
from collections import defaultdict, deque

def distanceK(root: 'TreeNode', target: 'TreeNode', K: int) -> List[int]:
    graph = defaultdict(list)

    # Build undirected graph from tree
    def build_graph(node, parent):
        if node and parent:
            graph[node.val].append(parent.val)
            graph[parent.val].append(node.val)
        if node.left:
            build_graph(node.left, node)
        if node.right:
            build_graph(node.right, node)

    build_graph(root, None)

    queue = deque([(target.val, 0)])
    seen = set([target.val])
    res = []

    while queue:
        node_val, dist = queue.popleft()
        if dist == K:
            res.append(node_val)
        elif dist < K:
            for neighbor in graph[node_val]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return res
```

---

## 937. Reorder Data in Log Files (Medium)

**Problem:** [LeetCode 937 - Reorder Data in Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

### Solution with Explanation

```python
def reorderLogFiles(logs: List[str]) -> List[str]:
    def is_digit_log(log):
        return log.split()[1].isdigit()

    def log_key(log):
        id_, rest = log.split(" ", 1)
        return (rest, id_)

    letter_logs = [log for log in logs if not is_digit_log(log)]
    digit_logs = [log for log in logs if is_digit_log(log)]

    # Sort letter-logs lexicographically by content, then identifier
    letter_logs.sort(key=log_key)
    return letter_logs + digit_logs
```

---

## 973. K Closest Points to Origin (Medium)

**Problem:** [LeetCode 973 - K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

### Solution with Explanation

```python
import heapq

def kClosest(points: List[List[int]], K: int) -> List[List[int]]:
    # Use a max-heap with negative distances to track K closest points
    heap = []

    for x, y in points:
        dist = -(x*x + y*y)  # negative because heapq is min-heap
        if len(heap) < K:
            heapq.heappush(heap, (dist, [x, y]))
        else:
            if dist > heap[0][0]:
                heapq.heapreplace(heap, (dist, [x, y]))

    return [point for _, point in heap]
```
# LeetCode Solutions with Explanations (Continued)

## 994. Rotting Oranges (Medium)

**Problem:** [LeetCode 994 - Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

### Solution with Explanation

```python
from collections import deque

def orangesRotting(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Initialize queue with rotten oranges and count fresh oranges
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    # Directions for adjacent cells
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    minutes_passed = 0

    # BFS to rot adjacent fresh oranges each minute
    while queue and fresh > 0:
        minutes_passed += 1
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc))

    return minutes_passed if fresh == 0 else -1
```

---

## 1152. Analyze User Website Visit Pattern (Medium)

**Problem:** [LeetCode 1152 - Analyze User Website Visit Pattern](https://leetcode.com/problems/analyze-user-website-visit-pattern/)

### Solution with Explanation

```python
from collections import defaultdict, Counter
from itertools import combinations

def mostVisitedPattern(username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    # Combine and sort by timestamp
    data = sorted(zip(timestamp, username, website))
    user_websites = defaultdict(list)

    # Build list of websites visited per user
    for _, user, site in data:
        user_websites[user].append(site)

    count = Counter()

    # For each user, find all unique 3-sequences
    for user, sites in user_websites.items():
        sequences = set(combinations(sites, 3))
        for seq in sequences:
            count[seq] += 1

    # Find the sequence with max count, lex smallest if tie
    max_count = max(count.values())
    candidates = [seq for seq, c in count.items() if c == max_count]
    return list(min(candidates))
```

---

## 1268. Search Suggestions System (Medium)

**Problem:** [LeetCode 1268 - Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/)

### Solution with Explanation

```python
from bisect import bisect_left

def suggestedProducts(products: List[str], searchWord: str) -> List[List[str]]:
    products.sort()
    result = []
    prefix = ""

    for ch in searchWord:
        prefix += ch
        # Find insertion position for prefix
        i = bisect_left(products, prefix)
        # Get up to 3 products starting with prefix
        suggestions = []
        for j in range(i, min(i + 3, len(products))):
            if products[j].startswith(prefix):
                suggestions.append(products[j])
        result.append(suggestions)

    return result
```

---

## 1291. Sequential Digits (Medium)

**Problem:** [LeetCode 1291 - Sequential Digits](https://leetcode.com/problems/sequential-digits/)

### Solution with Explanation

```python
def sequentialDigits(low: int, high: int) -> List[int]:
    result = []
    digits = "123456789"

    # Try all lengths from len(low) to len(high)
    for length in range(len(str(low)), len(str(high)) + 1):
        for start in range(0, 10 - length):
            num = int(digits[start:start+length])
            if low <= num <= high:
                result.append(num)

    return sorted(result)
```

---

## 1567. Maximum Length of Subarray With Positive Product (Medium)

**Problem:** [LeetCode 1567 - Maximum Length of Subarray With Positive Product](https://leetcode.com/problems/maximum-length-of-subarray-with-positive-product/)

### Solution with Explanation

```python
def getMaxLen(nums: List[int]) -> int:
    max_len = 0
    positive_len = 0
    negative_len = 0

    for num in nums:
        if num == 0:
            positive_len = 0
            negative_len = 0
        elif num > 0:
            positive_len += 1
            negative_len = negative_len + 1 if negative_len > 0 else 0
        else:
            prev_positive = positive_len
            positive_len = negative_len + 1 if negative_len > 0 else 0
            negative_len = prev_positive + 1
        max_len = max(max_len, positive_len)

    return max_len
```

# LeetCode Solutions with Explanations (Continued)

## 1628. Design an Expression Tree With Evaluate Function (Medium)

**Problem:** [LeetCode 1628 - Design an Expression Tree With Evaluate Function](https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/)

### Solution with Explanation

```python
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def evaluate(self) -> int:
        if self.val.isdigit():
            return int(self.val)
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()

        if self.val == '+':
            return left_val + right_val
        elif self.val == '-':
            return left_val - right_val
        elif self.val == '*':
            return left_val * right_val
        elif self.val == '/':
            return int(left_val / right_val)  # truncate towards zero

class ExpressionTreeBuilder:

    def buildTree(self, postfix: List[str]) -> 'Node':
        stack = []
        operators = {'+', '-', '*', '/'}
        for token in postfix:
            if token not in operators:
                stack.append(Node(token))
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(Node(token, left, right))
        return stack[-1]
```

---

## 1864. Minimum Number of Swaps to Make the Binary String Alternating (Medium)

**Problem:** [LeetCode 1864 - Minimum Number of Swaps to Make the Binary String Alternating](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/)

### Solution with Explanation

```python
def minSwaps(s: str) -> int:
    def check(pattern):
        swaps = 0
        for i, ch in enumerate(s):
            expected = pattern[i % 2]
            if ch != expected:
                swaps += 1
        return swaps // 2

    count0 = s.count('0')
    count1 = len(s) - count0

    # For alternating string, count0 and count1 difference can't be >1
    if abs(count0 - count1) > 1:
        return -1

    if count0 == count1:
        # Return min swaps for pattern starting with '0' or '1'
        return min(check('01'), check('10'))
    elif count0 > count1:
        return check('01')
    else:
        return check('10')
```

---

## 2055. Plates Between Candles (Medium)

**Problem:** [LeetCode 2055 - Plates Between Candles](https://leetcode.com/problems/plates-between-candles/)

### Solution with Explanation

```python
def platesBetweenCandles(s: str, queries: List[List[int]]) -> List[int]:
    n = len(s)
    prefix = [0] * (n + 1)
    left_candle = [-1] * n
    right_candle = [-1] * n

    # Prefix sum of plates
    for i in range(n):
        prefix[i+1] = prefix[i] + (1 if s[i] == '*' else 0)

    # Left nearest candle for each index
    prev = -1
    for i in range(n):
        if s[i] == '|':
            prev = i
        left_candle[i] = prev

    # Right nearest candle for each index
    prev = -1
    for i in range(n-1, -1, -1):
        if s[i] == '|':
            prev = i
        right_candle[i] = prev

    res = []
    for start, end in queries:
        left = right_candle[start]
        right = left_candle[end]
        if left == -1 or right == -1 or left >= right:
            res.append(0)
        else:
            res.append(prefix[right] - prefix[left])
    return res
```

---

## 2100. Find Good Days to Rob the Bank (Medium)

**Problem:** [LeetCode 2100 - Find Good Days to Rob the Bank](https://leetcode.com/problems/find-good-days-to-rob-the-bank/)

### Solution with Explanation

```python
def goodDaysToRobBank(security: List[int], time: int) -> List[int]:
    n = len(security)
    non_increasing = [0] * n
    non_decreasing = [0] * n

    # Count days non-increasing before i
    for i in range(1, n):
        if security[i] <= security[i-1]:
            non_increasing[i] = non_increasing[i-1] + 1

    # Count days non-decreasing after i
    for i in range(n-2, -1, -1):
        if security[i] <= security[i+1]:
            non_decreasing[i] = non_decreasing[i+1] + 1

    res = []
    for i in range(time, n - time):
        if non_increasing[i] >= time and non_decreasing[i] >= time:
            res.append(i)

    return res
```

---

## 2104. Sum of Subarray Ranges (Medium)

**Problem:** [LeetCode 2104 - Sum of Subarray Ranges](https://leetcode.com/problems/sum-of-subarray-ranges/)

### Solution with Explanation

```python
def subArrayRanges(nums: List[int]) -> int:
    n = len(nums)
    total = 0

    for i in range(n):
        min_val = max_val = nums[i]
        for j in range(i, n):
            min_val = min(min_val, nums[j])
            max_val = max(max_val, nums[j])
            total += max_val - min_val

    return total
```

# LeetCode Solutions with Explanations (Continued)

## 2214. Minimum Health to Beat Game (Medium)

**Problem:** [LeetCode 2214 - Minimum Health to Beat Game](https://leetcode.com/problems/minimum-health-to-beat-game/)

### Solution with Explanation

```python
def minimumHealth(damage: List[int], armor: int) -> int:
    max_damage = max(damage)
    total_damage = sum(damage)
    # Optimal to reduce the highest damage by armor (if possible)
    return total_damage - min(max_damage, armor) + 1
```

---

## 2221. Find Triangular Sum of an Array (Medium)

**Problem:** [LeetCode 2221 - Find Triangular Sum of an Array](https://leetcode.com/problems/find-triangular-sum-of-an-array/)

### Solution with Explanation

```python
def triangularSum(nums: List[int]) -> int:
    while len(nums) > 1:
        nums = [(nums[i] + nums[i+1]) % 10 for i in range(len(nums) - 1)]
    return nums[0]
```

---

## 2222. Number of Ways to Select Buildings (Medium)

**Problem:** [LeetCode 2222 - Number of Ways to Select Buildings](https://leetcode.com/problems/number-of-ways-to-select-buildings/)

### Solution with Explanation

```python
def numberOfWays(s: str) -> int:
    n = len(s)
    count0 = [0] * n
    count1 = [0] * n

    # Prefix counts of 0's and 1's
    for i in range(1, n):
        count0[i] = count0[i-1] + (s[i-1] == '0')
        count1[i] = count1[i-1] + (s[i-1] == '1')

    ways = 0
    for i in range(1, n-1):
        if s[i] == '0':
            # Patterns: 1 0 1
            ways += count1[i] * (count1[-1] - count1[i])
        else:
            # Patterns: 0 1 0
            ways += count0[i] * (count0[-1] - count0[i])
    return ways
```

---

## 2268. Minimum Number of Keypresses (Medium)

**Problem:** [LeetCode 2268 - Minimum Number of Keypresses](https://leetcode.com/problems/minimum-number-of-keypresses/)

### Solution with Explanation

```python
from collections import Counter

def minimumKeypresses(s: str) -> int:
    freq = sorted(Counter(s).values(), reverse=True)
    presses = 0

    for i, count in enumerate(freq):
        # Each letter can be typed 9 times per key (like old phones)
        # Cost is floor division by 9 + 1 times frequency
        presses += ((i // 9) + 1) * count

    return presses
```

---

## 2340. Minimum Adjacent Swaps to Make a Valid Array (Medium)

**Problem:** [LeetCode 2340 - Minimum Adjacent Swaps to Make a Valid Array](https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array/)

### Solution with Explanation

```python
def minimumAdjacentSwaps(nums: List[int]) -> int:
    n = len(nums)
    # Find all indices of 1s
    ones_positions = [i for i, num in enumerate(nums) if num == 1]
    median_index = len(ones_positions) // 2
    median_pos = ones_positions[median_index]

    swaps = 0
    for i, pos in enumerate(ones_positions):
        # Calculate swaps based on distance from median position adjusted by index
        swaps += abs(pos - (median_pos - median_index + i))

    return swaps
```

# LeetCode Solutions with Explanations (Continued)

## 42. Trapping Rain Water (Hard)

**Problem:** [LeetCode 42 - Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

### Solution with Explanation

```python
def trap(height: List[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    trapped = 0

    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            trapped += max(0, left_max - height[left])
        else:
            right -= 1
            right_max = max(right_max, height[right])
            trapped += max(0, right_max - height[right])

    return trapped
```

---

## 127. Word Ladder (Hard)

**Problem:** [LeetCode 127 - Word Ladder](https://leetcode.com/problems/word-ladder/)

### Solution with Explanation

```python
from collections import deque, defaultdict

def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    if endWord not in wordList:
        return 0

    L = len(beginWord)
    all_combo = defaultdict(list)

    for word in wordList:
        for i in range(L):
            all_combo[word[:i] + "*" + word[i+1:]].append(word)

    queue = deque([(beginWord, 1)])
    visited = {beginWord: True}

    while queue:
        current_word, level = queue.popleft()
        for i in range(L):
            intermediate = current_word[:i] + "*" + current_word[i+1:]
            for word in all_combo[intermediate]:
                if word == endWord:
                    return level + 1
                if word not in visited:
                    visited[word] = True
                    queue.append((word, level + 1))
            all_combo[intermediate] = []
    return 0
```

---

## 140. Word Break II (Hard)

**Problem:** [LeetCode 140 - Word Break II](https://leetcode.com/problems/word-break-ii/)

### Solution with Explanation

```python
def wordBreak(s: str, wordDict: List[str]) -> List[str]:
    word_set = set(wordDict)
    memo = {}

    def backtrack(start):
        if start in memo:
            return memo[start]

        results = []
        if start == len(s):
            results.append("")
            return results

        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for sub in backtrack(end):
                    results.append(word + ("" if sub == "" else " ") + sub)

        memo[start] = results
        return results

    return backtrack(0)
```

---

## 239. Sliding Window Maximum (Hard)

**Problem:** [LeetCode 239 - Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

### Solution with Explanation

```python
from collections import deque

def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    if not nums or k == 0:
        return []

    deq = deque()
    result = []

    for i, num in enumerate(nums):
        # Remove indexes outside the current window
        while deq and deq[0] <= i - k:
            deq.popleft()
        # Remove smaller numbers as they are not useful
        while deq and nums[deq[-1]] < num:
            deq.pop()

        deq.append(i)
        # Start adding to result once the first window is full
        if i >= k - 1:
            result.append(nums[deq[0]])

    return result
```

---

## 273. Integer to English Words (Hard)

**Problem:** [LeetCode 273 - Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)

### Solution with Explanation

```python
def numberToWords(num: int) -> str:
    if num == 0:
        return "Zero"

    below_20 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
                "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy",
            "Eighty", "Ninety"]
    thousands = ["", "Thousand", "Million", "Billion"]

    def helper(n):
        if n == 0:
            return ""
        elif n < 20:
            return below_20[n] + " "
        elif n < 100:
            return tens[n // 10] + " " + helper(n % 10)
        else:
            return below_20[n // 100] + " Hundred " + helper(n % 100)

    res = ""
    for i, unit in enumerate(thousands):
        if num % 1000 != 0:
            res = helper(num % 1000) + unit + " " + res
        num //= 1000

    return res.strip()
```