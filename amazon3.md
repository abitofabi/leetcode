# Amazon Data Architect Interview - Easy to Medium Python Questions & Solutions

## 1. Find Top K Frequent Elements

**Problem:** Given a list of elements, return the top K most frequent elements.

```python
from collections import Counter
import heapq
from typing import List

def topKFrequent(nums: List[int], k: int) -> List[int]:
    # Count frequency of each element
    count = Counter(nums)
    # Use a heap to get k elements with highest frequency
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

## 2. Merge Overlapping Intervals

**Problem:** Given a list of intervals, merge all overlapping ones.

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
        # Check overlap
        if current[0] <= prev[1]:
            # Merge intervals
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged
```

---

## 3. Sliding Window Sum

**Problem:** Calculate sum of every subarray/window of size k.

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

## 4. Count Unique Users in Time Window from Logs

**Problem:** Given log entries `(timestamp, user_id)`, count unique users in a given time interval.

```python
from typing import List, Tuple

def countUniqueUsers(logs: List[Tuple[int, str]], start: int, end: int) -> int:
    unique_users = set()
    for timestamp, user in logs:
        if start <= timestamp <= end:
            unique_users.add(user)
    return len(unique_users)
```

---

## 5. Detect Cycle in Job Dependency Graph

**Problem:** Given jobs and their dependencies, detect if a cycle exists.

```python
from collections import defaultdict
from typing import List

def hasCycle(numJobs: int, dependencies: List[List[int]]) -> bool:
    graph = defaultdict(list)
    for u, v in dependencies:
        graph[u].append(v)

    visited = [0] * numJobs  # 0=unvisited, 1=visiting, 2=visited

    def dfs(node):
        if visited[node] == 1:
            return True  # cycle found
        if visited[node] == 2:
            return False

        visited[node] = 1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        visited[node] = 2
        return False

    for job in range(numJobs):
        if visited[job] == 0:
            if dfs(job):
                return True
    return False
```

---

# Amazon Data Architect Interview - Easy to Medium Python Questions & Solutions (Continued)

## 6. Count Distinct Entries in a CSV Column (Memory-Efficient)

**Problem:** Given a CSV file and a column index, count distinct entries in that column without loading whole file.

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

## 7. Merge Sorted Lists of Timestamps

**Problem:** Merge multiple sorted lists into a single sorted list.

```python
import heapq
from typing import List

def mergeSortedLists(lists: List[List[int]]) -> List[int]:
    heap = []
    result = []

    # Initialize heap with first element of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, ele_idx = heapq.heappop(heap)
        result.append(val)
        if ele_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][ele_idx + 1], list_idx, ele_idx + 1))

    return result
```

---

## 8. Parse Logs and Extract Fields from Large JSON Lines File

**Problem:** Extract specific fields from a large JSON lines file memory-efficiently.

```python
import json
from typing import List

def extractFieldsFromLargeJSON(file_path: str, field_names: List[str]) -> List[dict]:
    extracted = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                extracted_record = {field: record.get(field, None) for field in field_names}
                extracted.append(extracted_record)
            except json.JSONDecodeError:
                continue  # skip bad lines
    return extracted
```

---

## 9. Top K Frequent Words

**Problem:** Given a list of words, return the k most frequent words sorted by frequency and lex order.

```python
from collections import Counter
import heapq
from typing import List

def topKFrequentWords(words: List[str], k: int) -> List[str]:
    count = Counter(words)
    heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(heap)

    result = []
    for _ in range(k):
        freq, word = heapq.heappop(heap)
        result.append(word)

    return result
```

---

## 10. Sliding Window Maximum (Medium)

**Problem:** Find maximum in every sliding window of size k.

```python
from collections import deque
from typing import List

def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    deq = deque()
    result = []

    for i, num in enumerate(nums):
        # Remove indices outside current window
        while deq and deq[0] <= i - k:
            deq.popleft()
        # Remove smaller values from the back
        while deq and nums[deq[-1]] < num:
            deq.pop()

        deq.append(i)

        if i >= k - 1:
            result.append(nums[deq[0]])

    return result
```

---
