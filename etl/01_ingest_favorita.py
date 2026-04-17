import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from config.db_config import JDBC_URL, DB_PROPERTIES


def main():

    spark = SparkSession.builder \
        .appName("01_Ingest_Favorita") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print("🔹 Starting Favorita ingestion...")

    # -----------------------------
    # 1️⃣ Load Raw Data
    # -----------------------------
    base_path = "data/raw/favorita/"

    items = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "items.csv")

    stores = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "stores.csv")

    sales = spark.read.option("header", True).option("inferSchema", True) \
        .csv(base_path + "train.csv")

    # -----------------------------
    # 2️⃣ Clean + Cast Sales
    # -----------------------------
    sales = sales.withColumn("date", to_date(col("date"))) \
        .withColumn("store_nbr", col("store_nbr").cast("int")) \
        .withColumn("item_nbr", col("item_nbr").cast("int")) \
        .withColumn("unit_sales", col("unit_sales").cast("double"))

    # Remove negative sales (returns)
    sales = sales.filter(col("unit_sales") >= 0)

    # -----------------------------
    # 3️⃣ Filter Perishable Items
    # -----------------------------
    perishable_items = items.filter(col("perishable") == 1)

    # -----------------------------
    # 4️⃣ Create Dimension Tables
    # -----------------------------
    dim_item = perishable_items.select(
        col("item_nbr").alias("item_id"),
        col("family"),
        col("class"),
        col("perishable").cast("boolean")
    ).dropDuplicates()

    dim_store = stores.select(
        col("store_nbr").alias("store_id"),
        col("city"),
        col("state"),
        col("type").alias("store_type"),
        col("cluster")
    ).dropDuplicates()

    # -----------------------------
    # 5️⃣ Create Fact Table
    # -----------------------------
    fact_sales = sales.join(
        perishable_items.select("item_nbr"),
        on="item_nbr",
        how="inner"
    )

    fact_sales = fact_sales.select(
        col("date"),
        col("store_nbr").alias("store_id"),
        col("item_nbr").alias("item_id"),
        col("unit_sales"),
        col("onpromotion").cast("boolean")
    )

    # -----------------------------
    # 6️⃣ Write to PostgreSQL
    # -----------------------------
    print("🔹 Writing tables to PostgreSQL...")

    dim_item.write.jdbc(
        url=JDBC_URL,
        table="retail.dim_item",
        mode="overwrite",
        properties=DB_PROPERTIES
    )

    dim_store.write.jdbc(
        url=JDBC_URL,
        table="retail.dim_store",
        mode="overwrite",
        properties=DB_PROPERTIES
    )

    fact_sales.write.jdbc(
        url=JDBC_URL,
        table="retail.fact_sales",
        mode="overwrite",
        properties=DB_PROPERTIES
    )

    # -----------------------------
    # 7️⃣ Save Parquet Snapshot
    # -----------------------------
    print("🔹 Saving parquet snapshot...")

    fact_sales.write.mode("overwrite") \
        .parquet("data/processed/parquet/fact_sales")

    print("✅ Ingestion complete.")

    spark.stop()


if __name__ == "__main__":
    main()