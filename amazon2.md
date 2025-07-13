# Amazon Data Engineer Python Coding Questions - Solutions

## 1. Find Top K Frequent Elements in a Large Dataset

**Problem:** Given a list of elements, return the top K most frequent elements.

```python
from collections import Counter
import heapq
from typing import List

def topKFrequent(nums: List[int], k: int) -> List[int]:
    # Count frequency of each element
    count = Counter(nums)
    # Use a heap to find k elements with highest frequency
    # heapq.nlargest returns k largest keys based on count
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

## 2. Merge Overlapping Intervals

**Problem:** Given a list of intervals [start, end], merge all overlapping intervals.

```python
from typing import List

def mergeIntervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            # Overlapping intervals, merge them
            prev[1] = max(prev[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)
    return merged
```

---

## 3. Parse Logs and Count Unique Users in a Time Window

**Problem:** Given a list of log entries with timestamp and user ID, count unique users within a time window [start, end].

```python
from typing import List, Tuple

def countUniqueUsers(logs: List[Tuple[int, str]], start: int, end: int) -> int:
    # logs: List of tuples (timestamp, user_id)
    unique_users = set()
    for timestamp, user in logs:
        if start <= timestamp <= end:
            unique_users.add(user)
    return len(unique_users)
```

---

## 4. Count Distinct Entries in a Column from Large CSV (Memory-Efficient)

**Problem:** Given a CSV file path and a column index, count distinct entries in that column without loading entire file in memory.

```python
import csv

def countDistinctColumnEntries(file_path: str, column_index: int) -> int:
    distinct_entries = set()
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > column_index:
                distinct_entries.add(row[column_index])
    return len(distinct_entries)
```

---

## 5. Rate Limiter (Token Bucket Algorithm)

**Problem:** Implement a rate limiter that allows up to `capacity` tokens, replenishing tokens at a certain `rate` per second.

```python
import time
from threading import Lock

class RateLimiter:
    def __init__(self, capacity: int, rate: float):
        self.capacity = capacity      # max tokens
        self.rate = rate              # tokens added per second
        self.tokens = capacity
        self.last_check = time.time()
        self.lock = Lock()

    def allow_request(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_check
            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_check = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False
```

---
# Amazon Data Engineer Python Coding Questions - Solutions (Continued)

## 6. Find Median from Data Stream

**Problem:** Implement a data structure that supports adding numbers and returning the median of all elements seen so far.

```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max heap for lower half (invert values to use Python's min heap as max heap)
        self.small = []
        # Min heap for upper half
        self.large = []

    def addNum(self, num: int) -> None:
        # Add to max heap (small)
        heapq.heappush(self.small, -num)
        # Balance: move largest of small to large
        if self.small and self.large and (-self.small[0]) > self.large[0]:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        # Maintain size property
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        else:
            return (-self.small[0] + self.large[0]) / 2
```

---

## 7. Simulate Sliding Window Aggregation (Sum or Count)

**Problem:** Implement a sliding window sum of size k over a list without using external libraries.

```python
from typing import List

def slidingWindowSum(nums: List[int], k: int) -> List[int]:
    n = len(nums)
    if n < k or k == 0:
        return []

    window_sum = sum(nums[:k])
    result = [window_sum]

    for i in range(k, n):
        window_sum += nums[i] - nums[i - k]
        result.append(window_sum)

    return result
```

---

## 8. Detect Cycle in Directed Acyclic Graph (DAG) of Job Dependencies

**Problem:** Given a list of jobs and dependencies, detect if a cycle exists.

```python
from collections import defaultdict

def hasCycle(numJobs: int, dependencies: List[List[int]]) -> bool:
    graph = defaultdict(list)
    for u, v in dependencies:
        graph[u].append(v)

    visited = [0] * numJobs  # 0 = unvisited, 1 = visiting, 2 = visited

    def dfs(node):
        if visited[node] == 1:
            return True  # cycle found
        if visited[node] == 2:
            return False  # already processed, no cycle here

        visited[node] = 1  # mark as visiting
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        visited[node] = 2  # mark as visited
        return False

    for job in range(numJobs):
        if visited[job] == 0:
            if dfs(job):
                return True
    return False
```

---

## 9. Merge Multiple Sorted Lists of Timestamps Efficiently

**Problem:** Given multiple sorted lists of timestamps, merge them into a single sorted timeline.

```python
import heapq
from typing import List

def mergeSortedLists(lists: List[List[int]]) -> List[int]:
    heap = []
    result = []

    # Initialize heap with first element from each list along with index info
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, ele_idx = heapq.heappop(heap)
        result.append(val)
        # If next element exists in same list, push it to heap
        if ele_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][ele_idx + 1], list_idx, ele_idx + 1))

    return result
```

---

## 10. Read Huge JSON Files Memory-Efficiently and Extract Fields

**Problem:** Read very large JSON file line-by-line and extract specific fields.

```python
import json

def extractFieldsFromLargeJSON(file_path: str, field_names: List[str]) -> List[dict]:
    extracted = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                extracted_record = {field: record.get(field, None) for field in field_names}
                extracted.append(extracted_record)
            except json.JSONDecodeError:
                # skip malformed JSON lines
                continue
    return extracted
```

---
# Amazon Data Engineer Python Coding Questions - Solutions (Continued)

## 11. LeetCode 692. Top K Frequent Words

**Problem:** Given a list of words, return the k most frequent words sorted by frequency and lex order.

```python
from collections import Counter
import heapq
from typing import List

def topKFrequentWords(words: List[str], k: int) -> List[str]:
    count = Counter(words)
    # Use a heap with (-freq, word) to get words by freq desc, then lex order asc
    heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(heap)
    result = []
    for _ in range(k):
        freq, word = heapq.heappop(heap)
        result.append(word)
    return result
```

---

## 12. LeetCode 56. Merge Intervals

**Problem:** Given a collection of intervals, merge all overlapping intervals.

```python
from typing import List

def mergeIntervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)

    return merged
```

---

## 13. LeetCode 23. Merge k Sorted Lists

**Problem:** Merge k sorted linked lists and return it as one sorted list.

```python
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    heap = []

    # Initialize heap with first node of each list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    curr = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

---

## 14. LeetCode 295. Find Median from Data Stream

**Problem:** Design a data structure supporting addNum and findMedian efficiently.

*(Solution already provided earlier — see problem 6)*

---

## 15. LeetCode 207. Course Schedule (Cycle Detection)

**Problem:** Given number of courses and prerequisites, determine if you can finish all courses.

```python
from collections import defaultdict

def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = defaultdict(list)
    for dest, src in prerequisites:
        graph[src].append(dest)

    visited = [0] * numCourses

    def dfs(node):
        if visited[node] == 1:
            return False  # cycle detected
        if visited[node] == 2:
            return True

        visited[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        visited[node] = 2
        return True

    for course in range(numCourses):
        if visited[course] == 0:
            if not dfs(course):
                return False
    return True
```
# Amazon Data Engineer Python Coding Questions - Solutions (Continued)

## 16. LeetCode 621. Task Scheduler

**Problem:** Given tasks represented by characters and a cooldown n, find the minimum time to finish all tasks.

```python
from collections import Counter

def leastInterval(tasks: List[str], n: int) -> int:
    task_counts = Counter(tasks)
    max_count = max(task_counts.values())
    max_count_tasks = list(task_counts.values()).count(max_count)

    # Calculate minimum intervals using formula:
    # (max_count - 1) * (n + 1) + max_count_tasks
    intervals = (max_count - 1) * (n + 1) + max_count_tasks
    return max(intervals, len(tasks))  # at least total tasks
```

---

## 17. LeetCode 148. Sort List (Merge Sort on Linked List)

**Problem:** Sort a linked list in O(n log n) time and constant space.

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sortList(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head

    # Find middle using slow and fast pointers
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sortList(head)
    right = sortList(mid)

    return merge(left, right)

def merge(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    tail = dummy

    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    tail.next = l1 if l1 else l2
    return dummy.next
```

---

## 18. Sliding Window Maximum (LeetCode 239)

**Problem:** (Repeat) Find maximum in every sliding window of size k.

*(Already provided above in problem 239 solution)*

---

## 19. Word Ladder (LeetCode 127)

**Problem:** (Repeat) Shortest transformation sequence from beginWord to endWord.

*(Already provided above in problem 127 solution)*

---

## 20. Word Break II (LeetCode 140)

**Problem:** (Repeat) All possible sentences from string and word dictionary.

*(Already provided above in problem 140 solution)*

---

If you'd like, I can continue with more or focus on specific problems or topics you prefer!