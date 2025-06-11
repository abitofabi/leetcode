SELECT product_name, year, price 
FROM Sales left join Product ON Sales.product_id = Product.product_id;