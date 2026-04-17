import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, trim, regexp_replace
from config.db_config import JDBC_URL, DB_PROPERTIES

def main():
    spark = SparkSession.builder.appName("02_Integrate_External_DEBUG").getOrCreate()

    print("JDBC_URL:", JDBC_URL)
    print("Reading fact_sales...")
    fact = spark.read.jdbc(JDBC_URL, "retail.fact_sales", properties=DB_PROPERTIES)
    print("fact_sales count =", fact.count())
    fact.show(5, truncate=False)

    print("Reading dim_store...")
    dim_store = spark.read.jdbc(JDBC_URL, "retail.dim_store", properties=DB_PROPERTIES)
    dim_store = dim_store.withColumn(
        "city_norm",
        lower(trim(regexp_replace(col("city"), r"\s+", " ")))
    ).select("store_id", "city_norm")
    print("dim_store count =", dim_store.count())
    dim_store.show(5, truncate=False)

    base_path = "Data/Raw/favorita/"

    print("Reading oil...")
    oil = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "oil.csv") \
        .withColumn("date", to_date(col("date"))) \
        .select("date", col("dcoilwtico").alias("oil_price"))
    print("oil count =", oil.count())
    oil.show(5, truncate=False)

    print("Reading transactions...")
    transactions = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "transactions.csv") \
        .withColumn("date", to_date(col("date"))) \
        .select(col("date"), col("store_nbr").alias("store_id"), col("transactions"))
    print("transactions count =", transactions.count())
    transactions.show(5, truncate=False)

    print("Reading holidays...")
    holidays = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "holidays_events.csv") \
        .withColumn("date", to_date(col("date"))) \
        .select("date").dropDuplicates() \
        .withColumn("is_holiday", col("date").isNotNull().cast("int"))
    print("holidays(unique dates) count =", holidays.count())
    holidays.show(5, truncate=False)

    print("Reading weather_daily parquet...")
    weather_daily = spark.read.parquet("Data/processed/parquet/weather_daily")
    print("weather_daily count =", weather_daily.count())
    weather_daily.show(5, truncate=False)

    # Join chain (all LEFT joins)
    enriched = fact \
        .join(oil, on="date", how="left") \
        .join(transactions, on=["date", "store_id"], how="left") \
        .join(holidays, on="date", how="left") \
        .join(dim_store, on="store_id", how="left") \
        .join(weather_daily, on=["date", "city_norm"], how="left")

    print("enriched count =", enriched.count())
    enriched.show(5, truncate=False)

    # Write (overwrite)
    enriched.write.jdbc(
        url=JDBC_URL,
        table="retail.fact_sales_enriched",
        mode="overwrite",
        properties=DB_PROPERTIES
    )

    print("✅ Wrote retail.fact_sales_enriched")
    spark.stop()

if __name__ == "__main__":
    main()