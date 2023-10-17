SELECT
  product_name,
  monday,
  max_price,
  y,
  y_lag_1,
  y_lag_2,
  y_lag_3,
  y_lag_4,
  y_lag_5,
  y_lag_6,
  (y_lag_1 + y_lag_2 + y_lag_3) / 3 AS y_avg_3,
  GREATEST(y_lag_1, y_lag_2, y_lag_3) AS y_max_3,
  LEAST(y_lag_1, y_lag_2, y_lag_3) AS y_min_3,
  (
    y_lag_1 + y_lag_2 + y_lag_3 + y_lag_4 + y_lag_5 + y_lag_6
  ) / 6 AS y_avg_6,
  GREATEST(
    y_lag_1,
    y_lag_2,
    y_lag_3,
    y_lag_4,
    y_lag_5,
    y_lag_6
  ) AS y_max_6,
  LEAST(
    y_lag_1,
    y_lag_2,
    y_lag_3,
    y_lag_4,
    y_lag_5,
    y_lag_6
  ) AS y_min_6,
  y_all_lag_1,
  y_all_lag_2,
  y_all_lag_3,
  y_all_lag_4,
  y_all_lag_5,
  y_all_lag_6,
  (y_all_lag_1 + y_all_lag_2 + y_all_lag_3) / 3 AS y_all_avg_3,
  GREATEST(y_all_lag_1, y_all_lag_2, y_all_lag_3) AS y_all_max_3,
  LEAST(y_all_lag_1, y_all_lag_2, y_all_lag_3) AS y_all_min_3,
  (
    y_all_lag_1 + y_all_lag_2 + y_all_lag_3 + y_all_lag_4 + y_all_lag_5 + y_all_lag_6
  ) / 6 AS y_all_avg_6,
  GREATEST(
    y_all_lag_1,
    y_all_lag_2,
    y_all_lag_3,
    y_all_lag_4,
    y_all_lag_5,
    y_all_lag_6
  ) AS y_all_max_6,
  LEAST(
    y_all_lag_1,
    y_all_lag_2,
    y_all_lag_3,
    y_all_lag_4,
    y_all_lag_5,
    y_all_lag_6
  ) AS y_all_min_6
FROM
  (
    SELECT
      *,
      SUM(y_lag_1) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_1,
      SUM(y_lag_2) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_2,
      SUM(y_lag_3) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_3,
      SUM(y_lag_4) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_4,
      SUM(y_lag_5) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_5,
      SUM(y_lag_6) OVER (
        PARTITION BY monday
        ORDER BY
          monday
      ) AS y_all_lag_6
    FROM
      (
        SELECT
          *,
          lagInFrame(y, 1) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_1,
          lagInFrame(y, 2) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_2,
          lagInFrame(y, 3) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_3,
          lagInFrame(y, 4) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_4,
          lagInFrame(y, 5) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_5,
          lagInFrame(y, 6) OVER (
            PARTITION BY product_name
            ORDER BY
              monday
          ) AS y_lag_6
        FROM
          (
            SELECT
              product_name,
              monday,
              MAX(price) AS max_price,
              COUNT(price) AS y
            FROM
              (
                SELECT
                  product_name,
                  DATE_TRUNC('week', dt) AS monday,
                  price
                FROM
                  default.data_sales_train
              ) AS t1
            GROUP BY
              product_name,
              monday
          ) AS t2
      ) AS t3
  ) AS t4
ORDER BY
  product_name,
  monday