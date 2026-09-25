create database superstore_sales;
USE superstore_sales;
SELECT DATABASE();
SET GLOBAL local_infile = 1;
CREATE TABLE orders (
    row_id INT,
    order_id VARCHAR(50),
    order_date VARCHAR(20),
    ship_date VARCHAR(20),
    ship_mode VARCHAR(50),
    customer_id VARCHAR(50),
    customer_name VARCHAR(100),
    segment VARCHAR(50),
    country VARCHAR(50),
    city VARCHAR(50),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    region VARCHAR(50),
    product_id VARCHAR(50),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    product_name VARCHAR(200),
    sales DECIMAL(10,2),
    quantity INT,
    discount DECIMAL(5,2),
    profit DECIMAL(10,4)
);
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/superstore_utf8.csv'
INTO TABLE orders
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;
DESCRIBE orders;
ALTER TABLE orders 
MODIFY sales DECIMAL(10,4),
MODIFY profit DECIMAL(10,4),
MODIFY discount DECIMAL(5,4);
SELECT 
MIN(sales), 
MAX(sales), 
AVG(sales) 
FROM orders;
SELECT `order_date`, `ship_date`
FROM orders
LIMIT 10;
ALTER TABLE orders
ADD COLUMN order_date_new DATE,
ADD COLUMN ship_date_new DATE;
UPDATE orders
SET order_date_new =
CASE 
    WHEN `order_date` LIKE '%/%'
        THEN STR_TO_DATE(`order_date`, '%m/%d/%Y')
    ELSE STR_TO_DATE(`order_date`, '%d-%m-%Y')
END;
SET SQL_SAFE_UPDATES = 0;
UPDATE orders
SET order_date_new =
CASE 
    WHEN `order_date` LIKE '%/%'
        THEN STR_TO_DATE(`order_date`, '%m/%d/%Y')
    ELSE STR_TO_DATE(`order_date`, '%d-%m-%Y')
END;
SET SQL_SAFE_UPDATES = 1;
UPDATE orders
SET ship_date_new =
CASE 
    WHEN `ship_date` LIKE '%/%'
        THEN STR_TO_DATE(`ship_date`, '%m/%d/%Y')
    ELSE STR_TO_DATE(`ship_date`, '%d-%m-%Y')
END;
SELECT ship_date, ship_date_new
FROM orders
LIMIT 10;

/* total sales for each month and year.*/
SELECT 
YEAR(`order_date_new`) AS year,
MONTH(`ship_date_new`) AS month,
SUM(sales)AS total_sales
FROM orders
GROUP BY year, month
ORDER BY year, month;

/*Calculate year-over-year sales growth*/

SELECT 
year,
total_sales,
LAG(total_sales) OVER (ORDER BY year) AS prev_year_sales,
ROUND(
((total_sales - LAG(total_sales) OVER (ORDER BY year)) 
/ LAG(total_sales) OVER (ORDER BY year)) * 100, 2
) AS growth_percentage
FROM (
    SELECT 
    YEAR(`order_date_new`) AS year,
    SUM(sales) AS total_sales
    FROM orders
    GROUP BY year
) t;

/*top 5 products by total profit.*/
SELECT 
product_name,
SUM(profit) AS total_profit
FROM orders
GROUP BY product_name
ORDER BY total_profit DESC
LIMIT 5;

/*total profit margin*/
SELECT 
ROUND((SUM(profit) / SUM(sales)) * 100, 2) 
AS profit_margin_percentage
FROM orders;

/* which category generates highest revenue?*/
SELECT 
category,
SUM(sales) AS total_sales
FROM orders
GROUP BY category
ORDER BY total_sales DESC;


