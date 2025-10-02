# ⚠️ SQL Execution Order Gotchas

These are the most common pitfalls interviewers love to test:

---

## 1. WHERE vs HAVING
- **WHERE** → filters rows before grouping (no aggregates allowed).  
- **HAVING** → filters groups after aggregation (aggregates allowed).  

❌ Wrong:
```sql
SELECT department_id, SUM(salary) AS total
FROM employees
GROUP BY department_id
HAVING total > 200000;   -- ❌ alias not visible yet
```

✅ Correct:
```sql
SELECT department_id, SUM(salary) AS total
FROM employees
GROUP BY department_id
HAVING SUM(salary) > 200000;
```

✅ Or with a CTE:
```sql
WITH dept_sums AS (
  SELECT department_id, SUM(salary) AS total
  FROM employees
  GROUP BY department_id
)
SELECT *
FROM dept_sums
WHERE total > 200000;
```

---

## 2. Filtering on Window Functions
- **WHERE** and **HAVING** happen before window functions.  
- You can’t filter directly on `RANK()`, `ROW_NUMBER()`, `LAG()`, etc.

❌ Wrong:
```sql
SELECT employee_id,
       RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
FROM employees
WHERE rnk <= 3;   -- ❌ rnk not defined yet
```

✅ Correct with CTE:
```sql
WITH ranked AS (
  SELECT employee_id,
         RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
  FROM employees
)
SELECT *
FROM ranked
WHERE rnk <= 3;
```

✅ Or with QUALIFY (Snowflake, BigQuery, Teradata, DuckDB):
```sql
SELECT employee_id,
       RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
FROM employees
QUALIFY rnk <= 3;
```

---

## 3. ORDER BY and Window Functions
- You *can* use window functions in `ORDER BY` because it runs **after** windows.

✅ Works:
```sql
SELECT employee_id,
       RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
FROM employees
ORDER BY rnk;
```

---

# 🧠 Execution Order (Simplified)

1. FROM  
2. WHERE  
3. GROUP BY  
4. HAVING  
5. WINDOW FUNCTIONS  
6. QUALIFY (if supported)  
7. SELECT  
8. ORDER BY  
9. LIMIT/TOP  

---

**Mnemonic:**  
👉 *Funny Wizards Grab Hats While Singing Opera Loudly*  
(FROM → WHERE → GROUP BY → HAVING → Window → SELECT → ORDER BY → LIMIT)  

---
