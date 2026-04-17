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
    spark = SparkSession.builder.appName("02B_Load_Feature_Tables").getOrCreate()

    base_path = "Data/Raw/favorita/"

    # -----------------------------
    # 1️⃣ Oil (Daily National)
    # -----------------------------
    oil = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "oil.csv"))
        .withColumn("date", to_date(col("date")))
        .select(
            col("date"),
            col("dcoilwtico").alias("oil_price")
        )
    )

    oil.write.format("jdbc") \
        .option("url", JDBC_URL) \
        .option("dbtable", "feat_oil_daily") \
        .option("user", DB_PROPERTIES["user"]) \
        .option("password", DB_PROPERTIES["password"]) \
        .option("driver", DB_PROPERTIES["driver"]) \
        .mode("overwrite") \
        .save()

    print("✅ Loaded feat_oil_daily")

    # -----------------------------
    # 2️⃣ Transactions (Daily Store)
    # -----------------------------
    transactions = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "transactions.csv"))
        .withColumn("date", to_date(col("date")))
        .select(
            col("date"),
            col("store_nbr").cast("int").alias("store_id"),
            col("transactions").cast("int")
        )
    )

    transactions.write.format("jdbc") \
        .option("url", JDBC_URL) \
        .option("dbtable", "feat_transactions_daily") \
        .option("user", DB_PROPERTIES["user"]) \
        .option("password", DB_PROPERTIES["password"]) \
        .option("driver", DB_PROPERTIES["driver"]) \
        .mode("overwrite") \
        .save()

    print("✅ Loaded feat_transactions_daily")

    # -----------------------------
    # 3️⃣ Holidays (Daily)
    # -----------------------------
    holidays = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(os.path.join(base_path, "holidays_events.csv"))
        .withColumn("date", to_date(col("date")))
        .select(
            col("date"),
            col("type").alias("holiday_type"),
            col("locale"),
            col("locale_name")
        )
        .dropDuplicates(["date"])
    )

    holidays.write.format("jdbc") \
        .option("url", JDBC_URL) \
        .option("dbtable", "feat_holidays_daily") \
        .option("user", DB_PROPERTIES["user"]) \
        .option("password", DB_PROPERTIES["password"]) \
        .option("driver", DB_PROPERTIES["driver"]) \
        .mode("overwrite") \
        .save()

    print("✅ Loaded feat_holidays_daily")

    # -----------------------------
    # 4️⃣ Weather (From Parquet)
    # -----------------------------
    weather = spark.read.parquet("Data/processed/parquet/weather_daily")

    weather = weather.withColumn("city_norm", normalize_city(col("city_norm")))

    weather.write.format("jdbc") \
        .option("url", JDBC_URL) \
        .option("dbtable", "feat_weather_daily_city") \
        .option("user", DB_PROPERTIES["user"]) \
        .option("password", DB_PROPERTIES["password"]) \
        .option("driver", DB_PROPERTIES["driver"]) \
        .mode("overwrite") \
        .save()

    print("✅ Loaded feat_weather_daily_city")

    spark.stop()


if __name__ == "__main__":
    main()