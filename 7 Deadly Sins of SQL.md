# ⚡ 7 Deadly Sins of SQL (Optimisation Edition)

## 1. Functions on Filter Columns (Non-Sargable)

❌ **Bad** – makes index useless
```sql
SELECT * 
FROM orders 
WHERE DATE(order_ts) = DATE('2025-01-01');
```

✅ **Good** – use range filter
```sql
SELECT * 
FROM orders 
WHERE order_ts >= TIMESTAMP('2025-01-01 00:00:00')
  AND order_ts <  TIMESTAMP('2025-01-02 00:00:00');
```

---

## 2. SELECT *

❌ **Bad** – fetches unnecessary data
```sql
SELECT * FROM customers;
```

✅ **Good** – fetch only what you need
```sql
SELECT customer_id, region 
FROM customers;
```

---

## 3. DISTINCT to Hide Duplicates

❌ **Bad** – DISTINCT after sloppy join
```sql
SELECT DISTINCT o.order_id
FROM orders o
JOIN order_items i ON o.order_id = i.order_id;
```

✅ **Good** – EXISTS avoids dup explosion
```sql
SELECT o.order_id
FROM orders o
WHERE EXISTS (
  SELECT 1 
  FROM order_items i 
  WHERE i.order_id = o.order_id
);
```

---

## 4. Correlated Subquery in SELECT

❌ **Bad** – runs per row (O(n²))
```sql
SELECT c.customer_id,
       (SELECT SUM(amount) 
        FROM payments p 
        WHERE p.customer_id = c.customer_id) AS total_payments
FROM customers c;
```

✅ **Good** – pre-aggregate + join once
```sql
WITH pay AS (
  SELECT customer_id, SUM(amount) AS total_payments
  FROM payments 
  GROUP BY customer_id
)
SELECT c.customer_id, p.total_payments
FROM customers c
LEFT JOIN pay p ON p.customer_id = c.customer_id;
```

---

## 5. Too Many Window Functions

❌ **Bad** – multiple scans of same partition
```sql
SELECT customer_id,
       SUM(amount) OVER (PARTITION BY customer_id) AS total_amt,
       COUNT(*)    OVER (PARTITION BY customer_id) AS order_cnt
FROM orders;
```

✅ **Good** – aggregate once and join back
```sql
WITH agg AS (
  SELECT customer_id, 
         SUM(amount) AS total_amt, 
         COUNT(*) AS order_cnt
  FROM orders
  GROUP BY customer_id
)
SELECT o.*, a.total_amt, a.order_cnt
FROM orders o
JOIN agg a USING (customer_id);
```

---

## 6. Joining Big Tables Before Aggregation

❌ **Bad** – join → explosion → aggregate
```sql
SELECT c.segment, SUM(s.amount)
FROM customers c
JOIN sales s ON c.customer_id = s.customer_id
GROUP BY c.segment;
```

✅ **Good** – aggregate first, join later
```sql
WITH sales_by_cust AS (
  SELECT customer_id, SUM(amount) AS amt
  FROM sales
  GROUP BY customer_id
)
SELECT c.segment, SUM(s.amt)
FROM customers c
JOIN sales_by_cust s ON c.customer_id = s.customer_id
GROUP BY c.segment;
```

---

## 7. IN Subquery vs EXISTS

❌ **Bad** – materialises subquery
```sql
SELECT c.customer_id
FROM customers c
WHERE c.customer_id IN (
  SELECT customer_id 
  FROM churn_list
);
```

✅ **Good** – EXISTS short-circuits
```sql
SELECT c.customer_id
FROM customers c
WHERE EXISTS (
  SELECT 1 
  FROM churn_list cl 
  WHERE cl.customer_id = c.customer_id
);
```

---

## ⚡ Quick Ref Mantra

- No **functions** on indexed columns  
- No **SELECT \***  
- No **DISTINCT** band-aids  
- No **correlated subqueries**  
- Reduce **window scans**  
- **Aggregate before join**  
- Prefer **EXISTS** over IN
