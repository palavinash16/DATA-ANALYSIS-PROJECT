CREATE DATABASE churn_db;
USE churn_db;
select count(*) from telco_churn;

SELECT * FROM telco_churn;

SELECT COUNT(*) AS total_customers FROM telco_churn;

SELECT DISTINCT Contract FROM telco_churn;

SELECT Churn, COUNT(*) AS count
FROM telco_churn
GROUP BY Churn;

SELECT 
    Churn,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telco_churn) AS percentage
FROM telco_churn
GROUP BY Churn;

SELECT Contract, Churn, COUNT(*) AS count
FROM telco_churn
GROUP BY Contract, Churn
ORDER BY Contract;

SELECT InternetService, Churn, COUNT(*) AS count
FROM telco_churn
GROUP BY InternetService, Churn;

SELECT PaymentMethod, Churn, COUNT(*) AS count
FROM telco_churn
GROUP BY PaymentMethod, Churn;

SELECT Churn, AVG(MonthlyCharges) AS avg_monthly
FROM telco_churn
GROUP BY Churn;

SELECT Churn, AVG(tenure) AS avg_tenure
FROM telco_churn
GROUP BY Churn;

SELECT 
    MAX(MonthlyCharges) AS max_charge,
    MIN(MonthlyCharges) AS min_charge
FROM telco_churn;

SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-1 Year'
        WHEN tenure <= 24 THEN '1-2 Years'
        WHEN tenure <= 48 THEN '2-4 Years'
        ELSE '4+ Years'
    END AS tenure_group,
    Churn,
    COUNT(*) AS count
FROM telco_churn
GROUP BY tenure_group, Churn
ORDER BY tenure_group;

SELECT SUM(TotalCharges) AS total_revenue
FROM telco_churn;

SELECT SUM(TotalCharges) AS churn_loss
FROM telco_churn
WHERE Churn = 'Yes';

SELECT Contract, InternetService, Churn, COUNT(*) AS count
FROM telco_churn
GROUP BY Contract, InternetService, Churn;

SELECT PaymentMethod, 
       COUNT(*) AS total,
       SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned
FROM telco_churn
GROUP BY PaymentMethod;

SELECT 
    (SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS churn_rate
FROM telco_churn;

SELECT *
FROM telco_churn
ORDER BY MonthlyCharges DESC
LIMIT 10;

SELECT *,
       AVG(MonthlyCharges) OVER (PARTITION BY Contract) AS avg_by_contract
FROM telco_churn;




