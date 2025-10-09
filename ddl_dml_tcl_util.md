# PostgreSQL SQL Cheat Sheet

## 🔹 DDL (Data Definition Language)
- **CREATE** → create objects  
  ```sql
  CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    salary NUMERIC(10,2)
  );
  ```
- **ALTER** → modify objects  
  ```sql
  ALTER TABLE employees ADD COLUMN department VARCHAR(50);
  ```
- **DROP** → remove objects  
  ```sql
  DROP TABLE employees;
  ```
- **TRUNCATE** → empty a table (faster than DELETE)  
  ```sql
  TRUNCATE TABLE employees;
  ```

---

## 🔹 DML (Data Manipulation Language)
- **SELECT** → read data  
  ```sql
  SELECT name, salary FROM employees WHERE salary > 50000;
  ```
- **INSERT** → add data  
  ```sql
  INSERT INTO employees (name, salary) VALUES ('Abi', 90000);
  ```
- **UPDATE** → modify data  
  ```sql
  UPDATE employees SET salary = 95000 WHERE name = 'Abi';
  ```
- **DELETE** → remove rows  
  ```sql
  DELETE FROM employees WHERE id = 5;
  ```

---

## 🔹 DCL (Data Control Language)
- **GRANT** → give permissions  
  ```sql
  GRANT SELECT, INSERT ON employees TO user1;
  ```
- **REVOKE** → remove permissions  
  ```sql
  REVOKE INSERT ON employees FROM user1;
  ```

---

## 🔹 TCL (Transaction Control Language)
- **BEGIN / START TRANSACTION** → start transaction  
- **COMMIT** → save changes  
- **ROLLBACK** → undo changes  
- **SAVEPOINT** → mark a point inside transaction  
- **RELEASE SAVEPOINT** → delete savepoint  
- **SET TRANSACTION** → set transaction properties  

Example:
```sql
BEGIN;
UPDATE employees SET salary = salary * 1.1;
SAVEPOINT before_bonus;
UPDATE employees SET salary = salary + 5000;
ROLLBACK TO SAVEPOINT before_bonus;
COMMIT;
```

---

## 🔹 Utility / Misc
- **EXPLAIN** → show query plan  
- **ANALYZE** → collect stats for optimizer  
- **VACUUM** → clean dead tuples  
- **INDEX** → speed up retrieval  
  ```sql
  CREATE INDEX idx_salary ON employees(salary);
  ```

---
