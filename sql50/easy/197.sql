SELECT X.id
FROM Weather AS X
JOIN Weather AS Y
  ON Y.recordDate = DATE_SUB(X.recordDate, INTERVAL 1 DAY)
WHERE X.temperature > Y.temperature;