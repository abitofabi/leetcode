# 📘 PostgreSQL SQL Interview Bible  

Your quick-reference guide for **scalar**, **aggregate**, and **window** functions with examples, results, and notes.  

---

## 🔤 Scalar Functions  

### String Functions  
| Function | Example | Result |  
|----------|---------|--------|  
| `upper(text)` | `upper('abinaya')` | `ABINAYA` |  
| `lower(text)` | `lower('Abi')` | `abi` |  
| `initcap(text)` | `initcap('hello world')` | `Hello World` |  
| `concat(a,b,...)` | `concat('Hello',' ','Abi')` | `Hello Abi` |  
| `substring(text, start, length)` | `substring('Abinaya',1,3)` | `Abi` |  
| `replace(text, from, to)` | `replace('2025-10-04','-','/')` | `2025/10/04` |  
| `trim(text)` | `trim('  Abi  ')` | `Abi` |  
| `lpad(text, len, fill)` | `lpad('7', 3, '0')` | `007` |  
| `rpad(text, len, fill)` | `rpad('7', 3, '0')` | `700` |  
| `length(text)` | `length('Abi')` | `3` |  

### Numeric & Math Functions  
| Function | Example | Result |  
|----------|---------|--------|  
| `abs(x)` | `abs(-42)` | `42` |  
| `round(x, n)` | `round(3.14159,2)` | `3.14` |  
| `ceil(x)` | `ceil(3.1)` | `4` |  
| `floor(x)` | `floor(3.9)` | `3` |  
| `power(a,b)` | `power(2,3)` | `8` |  
| `sqrt(x)` | `sqrt(16)` | `4` |  
| `mod(a,b)` | `mod(10,3)` | `1` |  
| `random()` | `random()` | `0.5481` (example) |  

### Date & Time Functions  
| Function | Example | Result |  
|----------|---------|--------|  
| `now()` | `now()` | `2025-10-04 17:30:00` |  
| `current_date` | `current_date` | `2025-10-04` |  
| `extract(field from date)` | `extract(year from now())` | `2025` |  
| `age(date1, date2)` | `age('2025-10-04','1992-01-01')` | `33 years 9 mons` |  
| `date_trunc(unit, date)` | `date_trunc('month', now())` | `2025-10-01 00:00:00` |  
| `make_date(y,m,d)` | `make_date(2025,10,4)` | `2025-10-04` |  

### Conditional Functions  
| Function | Example | Result |  
|----------|---------|--------|  
| `coalesce(a,b,...)` | `coalesce(null,'Abi')` | `Abi` |  
| `nullif(a,b)` | `nullif(5,5)` | `NULL` |  
| `greatest(a,b,c,...)` | `greatest(10,20,5)` | `20` |  
| `least(a,b,c,...)` | `least(10,20,5)` | `5` |  
| `case when ... then ... else ... end` | `case when salary>50000 then 'High' else 'Low' end` | `'High'` / `'Low'` |  

### Array Functions  
| Function | Example | Result |  
|----------|---------|--------|  
| `array_length(arr, dim)` | `array_length(array[10,20,30],1)` | `3` |  
| `cardinality(arr)` | `cardinality(array[1,2,3,4])` | `4` |  

---

## 📊 Aggregate Functions  

### Basic Aggregates  
| Function | Example | Result |  
|----------|---------|--------|  
| `count(*)` | `count(*)` | total rows |  
| `count(expr)` | `count(distinct user_id)` | unique user count |  
| `sum(expr)` | `sum(salary)` | sum of salaries |  
| `avg(expr)` | `avg(salary)` | average salary |  
| `min(expr)` | `min(age)` | smallest value |  
| `max(expr)` | `max(age)` | largest value |  

### String Aggregates  
| Function | Example | Result |  
|----------|---------|--------|  
| `string_agg(expr, delimiter)` | `string_agg(name, ', ')` | `Abi, Mike, Ian` |  
| `array_agg(expr)` | `array_agg(product_id)` | `{101,102,103}` |  

### Statistical Aggregates  
| Function | Example | Result |  
|----------|---------|--------|  
| `var_pop(expr)` | `var_pop(salary)` | population variance |  
| `stddev_pop(expr)` | `stddev_pop(salary)` | population stddev |  
| `percentile_cont(f) within group (order by col)` | `percentile_cont(0.5) within group (order by salary)` | median (continuous) |  
| `percentile_disc(f) within group (order by col)` | `percentile_disc(0.5) within group (order by salary)` | median (discrete) |  

---

## 📐 Window Functions  

### Ranking Functions  
| Function | Example | Result | Notes |  
|----------|---------|--------|-------|  
| `row_number()` | `row_number() over (partition by dept order by salary desc)` | 1,2,3… | Sequential, no gaps |  
| `rank()` | `rank() over (order by score desc)` | 1,2,2,4… | Gaps if ties |  
| `dense_rank()` | `dense_rank() over (order by score desc)` | 1,2,2,3… | No gaps |  
| `ntile(n)` | `ntile(4) over (order by salary desc)` | 1,2,3,4 quartiles | Divides into buckets |  

### Offset Functions  
| Function | Example | Result | Notes |  
|----------|---------|--------|-------|  
| `lag(expr, offset, default)` | `lag(salary,1) over (order by hire_date)` | previous row salary | Looks back |  
| `lead(expr, offset, default)` | `lead(salary,1) over (order by hire_date)` | next row salary | Looks forward |  
| `first_value(expr)` | `first_value(salary) over (partition by dept order by hire_date)` | first salary | |  
| `last_value(expr)` | `last_value(salary) over (partition by dept order by hire_date rows between unbounded preceding and unbounded following)` | last salary | Needs frame clause |  

### Aggregate as Window  
| Function | Example | Result | Notes |  
|----------|---------|--------|-------|  
| `sum(expr)` | `sum(sales) over (partition by region order by month)` | running total | |  
| `avg(expr)` | `avg(sales) over (partition by region order by month rows between 6 preceding and current row)` | rolling 7 months avg | |  
| `count(expr)` | `count(*) over (partition by dept)` | dept size per row | |  
| `min(expr)` | `min(salary) over (partition by dept)` | min salary per dept | |  
| `max(expr)` | `max(salary) over (partition by dept)` | max salary per dept | |  

### Frame Clauses  
- `rows between unbounded preceding and current row` → cumulative sum  
- `rows between 6 preceding and current row` → rolling 7 rows  
- `rows between current row and unbounded following` → forward looking  

### QUALIFY (if DB supports, e.g. Snowflake/BigQuery)  
```sql
select *,
       row_number() over (partition by dept order by salary desc) as rn
from employees
qualify rn <= 3;
```  

---

## ✨ Quick Rules of Thumb  
- **Scalar →** one value per row (string, math, date, array, conditional).  
- **Aggregate →** one value per group (`group by`).  
- **Window →** aggregate-like, but keeps all rows (adds context with `over()`).  
- **When stuck:**  
  - Ranking → top-N, leaderboards, deduplication.  
  - Offset → compare with previous/next row.  
  - Aggregate-as-window → running totals, moving averages.  

---

This cheat sheet covers **90% of SQL interview tasks**:  
1. String/date/numeric wrangling  
2. Aggregates & group analysis  
3. Window functions for ranking, trends, rolling metrics  


Biz cases:

---

## 🏢 Business Scenarios  

### 🎯 Customer Retention (Rolling Average)
```sql
select customer_id,
       avg(spend) over (
         partition by customer_id
         order by transaction_date
         rows between 6 preceding and current row
       ) as rolling_avg_7
from transactions;
```
👉 Identify VIPs whose spend is increasing every 6 months.  

---

### 🏆 Top-N Employees by Department
```sql
with ranked as (
  select employee_id, department_id, salary,
         dense_rank() over (partition by department_id order by salary desc) as rnk
  from employees
)
select * from ranked where rnk <= 3;
```
👉 Get top 3 paid employees in each department.  

---

### 📊 CTR Calculation
```sql
select app_id,
       round(100.0 * sum(case when event_type='click' then 1 else 0 end) /
             nullif(sum(case when event_type='impression' then 1 else 0 end),0),2) as ctr
from events
where extract(year from timestamp)=2022
group by app_id;
```
👉 Marketing KPI: Click-through rate per app.  

---

### 🛒 Basket Analysis (Product Combos)
```sql
select transaction_id, array_agg(product_id) as products
from transactions
group by transaction_id;
```
👉 See which products are often purchased together.  

---

### 📈 Salary Benchmarking
```sql
select distinct salary as second_highest_salary
from (
  select salary, dense_rank() over (order by salary desc) as rnk
  from employees
) t
where rnk = 2;
```
👉 HR: Find the **second highest salary** without duplicates.  

---

### 📅 Employee Performance Trend
```sql
select employee_id, month, sales,
       sales - lag(sales,1) over (partition by employee_id order by month) as diff_from_prev
from performance;
```
👉 See how each employee’s sales changed month-over-month.  

---
