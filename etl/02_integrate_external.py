import sys
import os

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, trim, regexp_replace
from config.db_config import JDBC_URL, DB_PROPERTIES


def normalize_city(col_expr):
    """
    Normalize city strings:
    - decode %20 to space
    - collapse whitespace
    - trim
    - lowercase
    """
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
        .appName("02_Integrate_External")
        # safer defaults for laptop; you can override via spark-submit --driver-memory
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    # -----------------------------
    # 1) Read base fact table (PARTITIONED JDBC READ)
    # -----------------------------
    # store_id is 1..54 in Favorita stores; perfect for JDBC partitioning
    print("🔹 Reading retail.fact_sales (partitioned JDBC)...")
    fact = spark.read.jdbc(
        url=JDBC_URL,
        table="retail.fact_sales",
        column="store_id",
        lowerBound=1,
        upperBound=54,
        numPartitions=8,
        properties=DB_PROPERTIES
    )

    # -----------------------------
    # 2) Read dim_store for city mapping
    # -----------------------------
    print("🔹 Reading retail.dim_store...")
    dim_store = spark.read.jdbc(
        url=JDBC_URL,
        table="retail.dim_store",
        properties=DB_PROPERTIES
    ).select("store_id", "city")

    dim_store = dim_store.withColumn("city_norm", normalize_city(col("city"))) \
                         .select("store_id", "city_norm")

    # -----------------------------
    # 3) Load Favorita external daily signals (CSV)
    # -----------------------------
    base_path = "Data/Raw/favorita/"

    print("🔹 Reading oil.csv ...")
    oil = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "oil.csv"))
        .withColumn("date", to_date(col("date")))
        .select("date", col("dcoilwtico").alias("oil_price"))
    )

    print("🔹 Reading transactions.csv ...")
    transactions = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "transactions.csv"))
        .withColumn("date", to_date(col("date")))
        .select(
            col("date"),
            col("store_nbr").cast("int").alias("store_id"),
            col("transactions").cast("int").alias("transactions")
        )
    )

    print("🔹 Reading holidays_events.csv ...")
    holidays = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "holidays_events.csv"))
        .withColumn("date", to_date(col("date")))
        .select("date")
        .dropDuplicates()
        .withColumn("is_holiday", col("date").isNotNull().cast("int"))
    )

    # -----------------------------
    # 4) Load preprocessed weather (parquet)
    # -----------------------------
    print("🔹 Reading weather_daily parquet ...")
    weather_daily = spark.read.parquet("Data/processed/parquet/weather_daily")

    # Ensure weather city_norm is normalized (handles %20)
    weather_daily = weather_daily.withColumn("city_norm", normalize_city(col("city_norm")))

    # -----------------------------
    # 5) Join everything (LEFT joins only)
    # -----------------------------
    print("🔹 Joining signals (left joins)...")
    enriched = (
        fact
        .join(oil, on="date", how="left")
        .join(transactions, on=["date", "store_id"], how="left")
        .join(holidays, on="date", how="left")
        .join(dim_store, on="store_id", how="left")
        .join(weather_daily, on=["date", "city_norm"], how="left")
    )

    # Fill missing values
    enriched = enriched.fillna({
        "oil_price": 0.0,
        "transactions": 0,
        "is_holiday": 0
    })

    # -----------------------------
    # 6) Write to Postgres + Parquet
    # -----------------------------
    print("🔹 Writing retail.fact_sales_enriched to PostgreSQL...")

    enriched.write \
        .format("jdbc") \
        .option("url", JDBC_URL) \
        .option("dbtable", "fact_sales_enriched") \
        .option("user", DB_PROPERTIES["user"]) \
        .option("password", DB_PROPERTIES["password"]) \
        .option("driver", DB_PROPERTIES["driver"]) \
        .option("batchsize", 5000) \
        .option("numPartitions", 8) \
        .option("isolationLevel", "NONE") \
        .mode("overwrite") \
        .save()

    print("🔹 Writing parquet snapshot...")
    enriched.write.mode("overwrite").parquet("Data/processed/parquet/fact_sales_enriched")

    print("✅ Done: fact_sales_enriched created.")
    spark.stop()


if __name__ == "__main__":
    main()