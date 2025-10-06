# 🧠 SQL Interview Cheatsheet for Data Engineering (Snowflake + PostgreSQL)

---

## 📊 1. SQL Schema Design

### ❖ Normalization Types

| Form | Description                          | Best Use                  |
| ---- | ------------------------------------ | ------------------------- |
| 1NF  | Atomic values, no repeating groups   | Basic normalization       |
| 2NF  | 1NF + remove partial dependencies    | For composite keys        |
| 3NF  | 2NF + remove transitive dependencies | Most widely used          |
| BCNF | Stronger 3NF                         | Very strict normalization |
| 4NF  | Remove multi-valued dependencies     | Rarely used in practice   |

### ❖ Star vs Snowflake Schema

| Feature          | Star Schema              | Snowflake Schema   |
| ---------------- | ------------------------ | ------------------ |
| Dimension Tables | Denormalized             | Normalized         |
| Joins            | Fewer                    | More               |
| Query Speed      | Faster                   | Slightly slower    |
| Storage          | More                     | Less               |
| Use Case         | BI Tools (e.g., Tableau) | Analytical Queries |

---

## ⏳ 2. SQL Query Execution Order

```
1. FROM
2. JOIN
3. WHERE
4. GROUP BY
5. HAVING
6. SELECT
7. DISTINCT
8. ORDER BY
9. LIMIT
```

---

## 🔄 3. Slowly Changing Dimensions (SCD)

| Type       | Description                         | Use Case           |
| ---------- | ----------------------------------- | ------------------ |
| SCD Type 1 | Overwrite old data                  | No history needed  |
| SCD Type 2 | Create new record with flag/version | Track full history |
| SCD Type 3 | Add new column for change           | Limited history    |

➡️ **In Snowflake**: Use `IS_CURRENT`, `EFFECTIVE_DATE`, and merge logic with `QUALIFY` or `ROW_NUMBER()` for SCD Type 2.

---

## 🪜 4. Partitioning & Window Functions

### Common Window Functions:

* `ROW_NUMBER()`
* `RANK()` / `DENSE_RANK()`
* `LAG()` / `LEAD()`
* `NTILE(n)`
* `SUM() OVER(...)`, `AVG() OVER(...)`

```sql
SELECT *, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rank
FROM employees;
```

---

## 🔍 5. Deduplication Techniques

```sql
-- Deduplicate keeping latest
SELECT *
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
  FROM users
) t
WHERE rn = 1;
```

---

## 🧩 6. PostgreSQL-Specific Functions

### Array & JSON

* `ARRAY_AGG()`
* `UNNEST()`
* `jsonb_extract_path()`, `jsonb_each()`

### Regex

* `~` (match regex)
* `~*` (case-insensitive match)
* `SIMILAR TO`, `REGEXP_REPLACE()`

---

## 🧮 7. Aggregates & Special Functions

* `SUM()`, `AVG()`, `MIN()`, `MAX()`, `COUNT()`
* `GROUPING SETS`, `ROLLUP`, `CUBE`
* `STRING_AGG()`, `ARRAY_AGG()`
* `CASE WHEN`, `COALESCE()`

---

## 📅 8. Date/Time Functions

### Snowflake & PostgreSQL

| Function                    | Purpose                  |
| --------------------------- | ------------------------ |
| `CURRENT_DATE`              | Today's date             |
| `DATE_TRUNC('month', date)` | Truncate to month        |
| `EXTRACT(DAY FROM date)`    | Extract part             |
| `DATE_PART('dow', date)`    | Day of week              |
| `AGE()` (Postgres)          | Difference between dates |

---

## 📈 9. Performance Optimization Techniques

### General

* Push filters early (`WHERE`/`JOIN` conditions)
* Use `LIMIT` during dev/debug
* Avoid `SELECT *`

### Snowflake

* Use **CLUSTER BY** for large tables
* Use **RESULT_SCAN** to reuse results
* Materialize intermediate steps via `CREATE TEMP TABLE`

### PostgreSQL

* Use **EXPLAIN ANALYZE** for query plan
* Proper **indexes** (B-tree, GIN, etc.)
* Vacuum frequently

### Informatica (SQ Filter Pushdown)

* Apply filter at **Source Qualifier** level
* Enable **Pushdown Optimization** to push logic to DB
* Reduce lookup and join bottlenecks

---

## ⚙️ 10. Indexing Strategies

| Type       | Best For                 |
| ---------- | ------------------------ |
| B-Tree     | Equality, Range searches |
| Hash       | Equality only            |
| GIN / GiST | Full-text or JSON search |
| Composite  | Multi-column filters     |

➡️ Use `EXPLAIN` to identify index usage.

---

## 🔗 11. JOIN Types & Use Cases

| JOIN Type  | Description                   | When to Use           |
| ---------- | ----------------------------- | --------------------- |
| INNER JOIN | Match rows in both            | Most common           |
| LEFT JOIN  | Keep all left rows            | Optional foreign data |
| RIGHT JOIN | Keep all right rows           | Rarely used           |
| FULL JOIN  | Keep all rows from both sides | Data reconciliation   |
| CROSS JOIN | Cartesian product             | Combinations          |
| SELF JOIN  | Join same table               | Hierarchies           |

---

## 🧠 12. Scaling Considerations

* Snowflake: Use **auto-scaling warehouses**, **multi-cluster warehouses**, **virtual warehouses**
* Partitioning: Split large tables into date-based or hash buckets
* Denormalization: Used for high-speed reads
* Materialized views / temp tables for reuse

---

## 📌 Bonus: SQL Coding Best Practices

* Comment complex logic
* Avoid deeply nested subqueries
* Alias tables clearly (`emp e`)
* Use `WITH` CTEs to modularize
* Avoid correlated subqueries inside `SELECT`
* Test on sample data

---

Would you like a version with **practice questions** next? Or a printable PDF? 😊
