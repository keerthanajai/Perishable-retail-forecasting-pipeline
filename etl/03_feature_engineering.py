import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit,
    year, month, dayofmonth, weekofyear, dayofweek,
    to_date, lower, trim, regexp_replace,
    lag, avg as spark_avg
)
from pyspark.sql.window import Window

from config.db_config import JDBC_URL, DB_PROPERTIES


def normalize_city(col_expr):
    return lower(
        trim(
            regexp_replace(
                regexp_replace(col_expr, "%20", " "),
                r"\s+",
                " "
            )
        )
    )


def main():

    spark = (
        SparkSession.builder
        .appName("03_Feature_Engineering")
        .config("spark.sql.shuffle.partitions", "48")
        .getOrCreate()
    )

    print("🔹 Reading fact_sales (partitioned JDBC)...")

    # -----------------------------
    # 1️⃣ Read base fact (partitioned)
    # -----------------------------
    fact = spark.read.jdbc(
        url=JDBC_URL,
        table="fact_sales",
        column="store_id",
        lowerBound=1,
        upperBound=54,
        numPartitions=8,
        properties=DB_PROPERTIES
    ).select("date", "store_id", "item_id", "unit_sales", "onpromotion")

    fact = fact.withColumn("date", to_date(col("date")))

    print("🔹 Reading dimensions...")

    # -----------------------------
    # 2️⃣ Read dimensions
    # -----------------------------
    dim_item = spark.read.jdbc(
        url=JDBC_URL,
        table="dim_item",
        properties=DB_PROPERTIES
    ).select("item_id", "family", "perishable")

    dim_store = spark.read.jdbc(
        url=JDBC_URL,
        table="dim_store",
        properties=DB_PROPERTIES
    ).select("store_id", "city")

    dim_store = dim_store.withColumn(
        "city_norm",
        normalize_city(col("city"))
    ).select("store_id", "city_norm")

    # Robust perishable filter
    fact = fact.join(dim_item, on="item_id", how="inner").filter(
        (col("perishable") == True) |
        (col("perishable") == 1) |
        (col("perishable") == "1")
    )

    base = fact.join(dim_store, on="store_id", how="left")

    print("🔹 Reading feature tables...")

    # -----------------------------
    # 3️⃣ Feature tables
    # -----------------------------
    oil = spark.read.jdbc(
        url=JDBC_URL,
        table="feat_oil_daily",
        properties=DB_PROPERTIES
    ).withColumn("date", to_date(col("date"))) \
     .select("date", "oil_price")

    trx = spark.read.jdbc(
        url=JDBC_URL,
        table="feat_transactions_daily",
        properties=DB_PROPERTIES
    ).withColumn("date", to_date(col("date"))) \
     .select("date", "store_id", "transactions")

    hol = spark.read.jdbc(
        url=JDBC_URL,
        table="feat_holidays_daily",
        properties=DB_PROPERTIES
    ).withColumn("date", to_date(col("date"))) \
     .select("date", "holiday_type", "locale", "locale_name") \
     .withColumn("is_holiday", lit(1))

    weather = spark.read.jdbc(
        url=JDBC_URL,
        table="feat_weather_daily_city",
        properties=DB_PROPERTIES
    ).withColumn("date", to_date(col("date"))) \
     .withColumn("city_norm", normalize_city(col("city_norm")))

    print("🔹 Joining features...")

    # -----------------------------
    # 4️⃣ Join everything (LEFT joins)
    # -----------------------------
    df = (
        base
        .join(oil, on="date", how="left")
        .join(trx, on=["date", "store_id"], how="left")
        .join(hol, on="date", how="left")
        .join(weather, on=["date", "city_norm"], how="left")
    )

    df = df.fillna({
        "oil_price": 0.0,
        "transactions": 0,
        "is_holiday": 0
    })

    print("🔹 Creating calendar features...")

    # -----------------------------
    # 5️⃣ Calendar features
    # -----------------------------
    df = df.withColumn("year", year(col("date"))) \
           .withColumn("month", month(col("date"))) \
           .withColumn("day", dayofmonth(col("date"))) \
           .withColumn("weekofyear", weekofyear(col("date"))) \
           .withColumn("dow", dayofweek(col("date"))) \
           .withColumn("is_weekend", when(col("dow").isin([1, 7]), 1).otherwise(0)) \
           .withColumn("is_month_start", when(col("day") == 1, 1).otherwise(0)) \
           .withColumn("is_month_end", when(col("day") >= 28, 1).otherwise(0))

    # Reduce columns BEFORE window operations
    df = df.select(
        "date", "store_id", "item_id", "unit_sales", "onpromotion",
        "family",
        "oil_price", "transactions", "is_holiday",
        "temp_avg_c", "temp_min_c", "temp_max_c",
        "precip_total_mm", "rain_total_mm",
        "humidity_avg_pct", "apparent_avg_c",
        "year", "month", "day", "weekofyear", "dow",
        "is_weekend", "is_month_start", "is_month_end"
    )

    print("🔹 Computing lag + rolling features...")

    # -----------------------------
    # 6️⃣ Lag + rolling
    # -----------------------------
    df = df.repartition(64, "store_id", "item_id")
    df = df.persist()

    w = Window.partitionBy("store_id", "item_id").orderBy(col("date"))

    df = df.withColumn("lag_1", lag(col("unit_sales"), 1).over(w)) \
           .withColumn("lag_7", lag(col("unit_sales"), 7).over(w)) \
           .withColumn("lag_14", lag(col("unit_sales"), 14).over(w)) \
           .withColumn("lag_28", lag(col("unit_sales"), 28).over(w))

    w7  = w.rowsBetween(-7, -1)
    w14 = w.rowsBetween(-14, -1)
    w28 = w.rowsBetween(-28, -1)

    df = df.withColumn("roll_mean_7",  spark_avg(col("unit_sales")).over(w7)) \
           .withColumn("roll_mean_14", spark_avg(col("unit_sales")).over(w14)) \
           .withColumn("roll_mean_28", spark_avg(col("unit_sales")).over(w28))

    # Remove early rows without history
    df = df.filter(col("lag_28").isNotNull())

    print("🔹 Finalizing model dataset...")

    model_df = df.select(
        "date", "store_id", "item_id",
        col("unit_sales").alias("y"),
        "onpromotion", "family",
        "oil_price", "transactions", "is_holiday",
        "temp_avg_c", "temp_min_c", "temp_max_c",
        "precip_total_mm", "rain_total_mm",
        "humidity_avg_pct", "apparent_avg_c",
        "year", "month", "weekofyear", "dow",
        "is_weekend", "is_month_start", "is_month_end",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28"
    )

    out_path = "Data/processed/feature_tables/model_train.parquet"
    os.makedirs("Data/processed/feature_tables", exist_ok=True)

    model_df.write.mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(out_path)

    print(f"✅ Model dataset written to {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()