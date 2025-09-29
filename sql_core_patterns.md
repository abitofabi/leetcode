# 📘 SQL Interview Core Patterns – Playbook

This playbook covers the **5 must-know SQL interview patterns**, with:
- Business-style problem statements  
- Query templates  
- Sample datasets + expected outputs  
- Edge case notes  

---

## 🧩 Pattern #1 – Top-N per Group (Ranking)

**Business Question:**  
*"Find the top 3 highest-paid employees in each department."*

**Schema:**  
```sql
employees (
  employee_id   INT PRIMARY KEY,
  name          VARCHAR,
  department_id INT,
  salary        DECIMAL
)
```

**Sample Data:**
| employee_id | name   | department_id | salary |
|-------------|--------|---------------|--------|
| 1           | Alice  | 10            | 120000 |
| 2           | Bob    | 10            | 115000 |
| 3           | Carol  | 10            | 115000 |
| 4           | Dave   | 20            | 90000  |
| 5           | Erin   | 20            | 85000  |

**Query:**
```sql
WITH ranked_employees AS (
    SELECT 
        employee_id,
        name,
        department_id,
        salary,
        RANK() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS rnk
    FROM employees
)
SELECT *
FROM ranked_employees
WHERE rnk <= 3;
```

**Expected Output:**
| employee_id | name  | department_id | salary | rnk |
|-------------|-------|---------------|--------|-----|
| 1           | Alice | 10            | 120000 | 1   |
| 2           | Bob   | 10            | 115000 | 2   |
| 3           | Carol | 10            | 115000 | 2   |
| 4           | Dave  | 20            | 90000  | 1   |
| 5           | Erin  | 20            | 85000  | 2   |

**Edge Cases:**  
- Use `ROW_NUMBER()` if you want exactly 3 rows per dept.  
- Use `DENSE_RANK()` if you want ties included.  

---

## 🧩 Pattern #2 – Nth Row / First & Last Record

**Business Question:**  
*"For each customer, find their very first order (date + amount)."*

**Schema:**  
```sql
orders (
  order_id    INT PRIMARY KEY,
  customer_id INT,
  order_date  DATE,
  amount      DECIMAL
)
```

**Sample Data:**
| order_id | customer_id | order_date  | amount |
|----------|-------------|-------------|--------|
| 1        | 101         | 2024-01-05  | 200    |
| 2        | 101         | 2024-02-10  | 150    |
| 3        | 102         | 2024-01-20  | 300    |
| 4        | 102         | 2024-01-25  | 400    |

**Query:**
```sql
WITH ordered AS (
    SELECT
        customer_id,
        order_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY order_date ASC
        ) AS rn
    FROM orders
)
SELECT *
FROM ordered
WHERE rn = 1;
```

**Expected Output:**
| customer_id | order_id | order_date | amount | rn |
|-------------|----------|------------|--------|----|
| 101         | 1        | 2024-01-05 | 200    | 1  |
| 102         | 3        | 2024-01-20 | 300    | 1  |

**Edge Cases:**  
- Last order → change `ASC` → `DESC`.  
- 2nd order → use `WHERE rn = 2`.  

---

## 🧩 Pattern #3 – Comparisons Across Rows

**Business Question:**  
*"Find months where a customer’s spend increased compared to the previous month."*

**Schema:**  
```sql
orders (
  order_id    INT PRIMARY KEY,
  customer_id INT,
  order_date  DATE,
  amount      DECIMAL
)
```

**Sample Data:**
| order_id | customer_id | order_date  | amount |
|----------|-------------|-------------|--------|
| 1        | 201         | 2024-01-10  | 100    |
| 2        | 201         | 2024-02-10  | 200    |
| 3        | 201         | 2024-03-10  | 150    |

**Query:**
```sql
WITH monthly_spend AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_spend
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
with_lag AS (
    SELECT
        customer_id,
        month,
        total_spend,
        LAG(total_spend) OVER (
            PARTITION BY customer_id
            ORDER BY month
        ) AS prev_spend
    FROM monthly_spend
)
SELECT *
FROM with_lag
WHERE total_spend > prev_spend;
```

**Expected Output:**
| customer_id | month      | total_spend | prev_spend |
|-------------|------------|-------------|------------|
| 201         | 2024-02-01 | 200         | 100        |

**Edge Cases:**  
- First month will have `NULL` in `prev_spend`.  
- Use `LEAD()` to compare with next row instead.  

---

## 🧩 Pattern #4 – Running Totals & Moving Averages

**Business Question:**  
*"Compute cumulative spend per customer over time."*

**Schema:**  
```sql
orders (
  order_id    INT PRIMARY KEY,
  customer_id INT,
  order_date  DATE,
  amount      DECIMAL
)
```

**Sample Data:**
| order_id | customer_id | order_date  | amount |
|----------|-------------|-------------|--------|
| 1        | 301         | 2024-01-01  | 100    |
| 2        | 301         | 2024-01-05  | 200    |
| 3        | 301         | 2024-01-10  | 50     |

**Query:**
```sql
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM orders;
```

**Expected Output:**
| customer_id | order_date  | amount | running_total |
|-------------|-------------|--------|---------------|
| 301         | 2024-01-01  | 100    | 100           |
| 301         | 2024-01-05  | 200    | 300           |
| 301         | 2024-01-10  | 50     | 350           |

**Edge Cases:**  
- For rolling 7-day average:  
```sql
AVG(amount) OVER (
    PARTITION BY customer_id
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
) AS rolling_avg_7d
```

---

## 🧩 Pattern #5 – Streaks / Consecutive Activity (Gaps & Islands)

**Business Question:**  
*"Find customers with at least 3 consecutive months of increasing spend."*

**Schema:**  
```sql
orders (
  order_id    INT PRIMARY KEY,
  customer_id INT,
  order_date  DATE,
  amount      DECIMAL
)
```

**Sample Data:**
| order_id | customer_id | order_date  | amount |
|----------|-------------|-------------|--------|
| 1        | 401         | 2024-01-01  | 100    |
| 2        | 401         | 2024-02-01  | 200    |
| 3        | 401         | 2024-03-01  | 300    |
| 4        | 401         | 2024-04-01  | 250    |

**Query:**
```sql
WITH monthly_spend AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_spend
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
with_lag AS (
    SELECT
        customer_id,
        month,
        total_spend,
        LAG(total_spend) OVER (
            PARTITION BY customer_id
            ORDER BY month
        ) AS prev_spend
    FROM monthly_spend
),
flagged AS (
    SELECT
        customer_id,
        month,
        CASE WHEN total_spend > prev_spend THEN 1 ELSE 0 END AS is_increase
    FROM with_lag
),
grouped AS (
    SELECT
        customer_id,
        month,
        is_increase,
        SUM(CASE WHEN is_increase = 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY customer_id ORDER BY month) AS grp
    FROM flagged
),
streaks AS (
    SELECT customer_id, grp, COUNT(*) AS streak_len
    FROM grouped
    WHERE is_increase = 1
    GROUP BY customer_id, grp
)
SELECT DISTINCT customer_id
FROM streaks
WHERE streak_len >= 3;
```

**Expected Output:**
| customer_id |
|-------------|
| 401         |

**Edge Cases:**  
- Missing months break the streak (need a calendar table to fix).  
- Change `>= 3` to any streak length.  

---

# ✅ Summary
- **Top-N per group** → `RANK()` / `DENSE_RANK()`  
- **Nth row (first/last)** → `ROW_NUMBER()`  
- **Comparisons across rows** → `LAG()` / `LEAD()`  
- **Running totals / rolling averages** → `SUM()` / `AVG() OVER`  
- **Streaks / consecutive activity** → `ROW_NUMBER()` trick or grouping logic  

---
