-- Q7: Daily Running Revenue

WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_total
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)

SELECT
    order_date,
    daily_total,
    SUM(daily_total) OVER (
        ORDER BY order_date
    ) AS running_revenue
FROM daily_revenue
ORDER BY order_date;