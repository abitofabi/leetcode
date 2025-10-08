# ⚡ PostgreSQL 1-Page Crib Sheet  

## 🔤 Scalar  
- **Strings**: `upper()`, `lower()`, `initcap()`, `concat()`, `substring()`, `replace()`, `trim()`, `lpad()/rpad()`, `length()`  
- **Math**: `abs()`, `round(x,n)`, `ceil()`, `floor()`, `power()`, `sqrt()`, `mod()`, `random()`  
- **Date/Time**: `now()`, `current_date`, `extract(field from ts)`, `age(d1,d2)`, `date_trunc('month',ts)`, `make_date(y,m,d)`  
- **Conditional**: `coalesce()`, `nullif()`, `greatest()`, `least()`, `case when ... end`  

---

## 📊 Aggregates  
- `count(*)`, `count(distinct x)`  
- `sum(x)`, `avg(x)`, `min(x)`, `max(x)`  
- `string_agg(expr, ',')`, `array_agg(expr)`  
- `percentile_cont(0.5) within group (order by x)` → median  

---

## 📐 Window  
- **Ranking**:  
  - `row_number()` → dedupe  
  - `rank()` → 1,2,2,4 (gaps)  
  - `dense_rank()` → 1,2,2,3 (no gaps)  
  - `ntile(n)` → quartiles/deciles  
- **Offset**: `lag()`, `lead()`, `first_value()`, `last_value()`  
- **Running/rolling**: `sum()/avg()/count() over (...)`  

---

## 🏢 Scenarios  
- **Top-N per group**: `dense_rank() over (partition by dept order by salary desc)`  
- **CTR**: `sum(click)/nullif(sum(impression),0)`  
- **Median salary**: `percentile_cont(0.5) within group (order by salary)`  
- **Rolling avg spend**: `avg(spend) over (partition by cust order by date rows 6 preceding)`  
- **Basket analysis**: `array_agg(product_id) group by transaction_id`  
- **Salary benchmarking**: 2nd highest → `dense_rank() = 2`  

---

✅ **Rule of thumb**  
- **Scalar** → row-level ops  
- **Aggregate** → collapse rows  
- **Window** → add context to each row  
