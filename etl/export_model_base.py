import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, trim, regexp_replace
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
    spark = SparkSession.builder.appName("Export_Model_Base").getOrCreate()

    fact = spark.read.jdbc(
        url=JDBC_URL,
        table="fact_sales",
        column="store_id",
        lowerBound=1,
        upperBound=54,
        numPartitions=8,
        properties=DB_PROPERTIES
    ).withColumn("date", to_date(col("date")))

    dim_item = spark.read.jdbc(JDBC_URL, "dim_item", properties=DB_PROPERTIES)
    dim_store = spark.read.jdbc(JDBC_URL, "dim_store", properties=DB_PROPERTIES)

    dim_store = dim_store.withColumn("city_norm", normalize_city(col("city")))

    oil = spark.read.jdbc(JDBC_URL, "feat_oil_daily", properties=DB_PROPERTIES)
    trx = spark.read.jdbc(JDBC_URL, "feat_transactions_daily", properties=DB_PROPERTIES)
    hol = spark.read.jdbc(JDBC_URL, "feat_holidays_daily", properties=DB_PROPERTIES)
    weather = spark.read.jdbc(JDBC_URL, "feat_weather_daily_city", properties=DB_PROPERTIES)

    df = (
        fact
        .join(dim_item, "item_id", "inner")
        .filter(col("perishable") == True)
        .join(dim_store.select("store_id", "city_norm"), "store_id", "left")
        .join(oil, "date", "left")
        .join(trx, ["date", "store_id"], "left")
        .join(hol, "date", "left")
        .join(weather, ["date", "city_norm"], "left")
    )

    df.write.mode("overwrite").parquet("Data/processed/model_base.parquet")

    spark.stop()


if __name__ == "__main__":
    main()