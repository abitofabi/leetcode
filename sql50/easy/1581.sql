SELECT customer_id, COUNT(*) count_no_trans
FROM Visits LEFT JOIN Transactions ON Visits.visit_id=Transactions.visit_id
WHERE transaction_id is NULL
GROUP BY customer_id; 