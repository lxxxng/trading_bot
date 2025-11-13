-- SQLite
SELECT
    strftime('%Y-%m', close_ts_utc)                    AS year_month,
    COUNT(*)                                           AS total_trades,
    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END)    AS wins,
    ROUND(100.0 * SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                       AS win_rate_pct,
    SUM(pl)                                            AS net_pl
FROM trades
GROUP BY strftime('%Y-%m', close_ts_utc)
ORDER BY year_month;