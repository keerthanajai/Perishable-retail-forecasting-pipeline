import sys, os, glob, csv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, to_date, lower, trim, regexp_replace,
    avg, min as spark_min, max as spark_max, sum as spark_sum
)

RAW_WEATHER_DIR = "Data/Raw/weather"
CLEAN_WEATHER_DIR = "Data/processed/intermediate/weather_clean"
OUT_PARQUET_DIR = "Data/processed/parquet/weather_daily"

def clean_one_file(in_path: str, out_path: str):
    """
    Weather files have 3 junk lines (metadata + blank) before the real header line.
    We remove those and keep only the time-series CSV part.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the real header line that starts with "time"
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find 'time' header in {in_path}")

    ts_lines = lines[header_idx:]  # from header to end

    # Write cleaned CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as out_f:
        out_f.writelines(ts_lines)

def main():
    # 1) Clean raw weather files into standard CSVs
    os.makedirs(CLEAN_WEATHER_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_WEATHER_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No weather CSVs found in {RAW_WEATHER_DIR}")

    print(f"🔹 Found {len(files)} weather files. Cleaning headers...")

    for in_file in files:
        city = Path(in_file).stem  # filename without extension
        out_file = os.path.join(CLEAN_WEATHER_DIR, f"{city}.csv")
        clean_one_file(in_file, out_file)

    print(f"✅ Cleaned files written to: {CLEAN_WEATHER_DIR}")

    # 2) Spark: read cleaned CSVs and aggregate hourly -> daily
    spark = SparkSession.builder.appName("02A_Preprocess_Weather").getOrCreate()

    w = spark.read \
        .option("header", True) \
        .option("inferSchema", True) \
        .csv(os.path.join(CLEAN_WEATHER_DIR, "*.csv"))

    # Add city from filename by reading each file separately? Easiest:
    # We'll re-read with wholeTextFiles is heavy.
    # Instead, we will add city by mapping using input_file_name().
    from pyspark.sql.functions import input_file_name
    w = w.withColumn("file_path", input_file_name())
    w = w.withColumn(
        "city",
        regexp_replace(regexp_replace(col("file_path"), r"^.*\/", ""), r"\.csv$", "")
    )
    w = w.withColumn(
        "city_norm",
        lower(
            trim(
                regexp_replace(
                    regexp_replace(col("city"), "%20", " "),
                    r"\s+",
                    " "
                )
            )
        )
    )

    # Parse time -> date
    # Your time format is like "2013-01-01 0:00" (single digit hour)
    w = w.withColumn("ts", to_timestamp(col("time"), "yyyy-MM-dd H:mm"))
    w = w.withColumn("date", to_date(col("ts")))

    # Rename columns (Spark can read them, but we keep them clean)
    w = w.select(
        "date", "city_norm",
        col("temperature_2m (°C)").alias("temp_c"),
        col("precipitation (mm)").alias("precip_mm"),
        col("rain (mm)").alias("rain_mm"),
        col("relative_humidity_2m (%)").alias("rh_pct"),
        col("apparent_temperature (°C)").alias("apparent_c"),
    )

    weather_daily = w.groupBy("date", "city_norm").agg(
        avg("temp_c").alias("temp_avg_c"),
        spark_min("temp_c").alias("temp_min_c"),
        spark_max("temp_c").alias("temp_max_c"),
        spark_sum("precip_mm").alias("precip_total_mm"),
        spark_sum("rain_mm").alias("rain_total_mm"),
        avg("rh_pct").alias("humidity_avg_pct"),
        avg("apparent_c").alias("apparent_avg_c"),
    )

    os.makedirs(OUT_PARQUET_DIR, exist_ok=True)
    weather_daily.write.mode("overwrite").parquet(OUT_PARQUET_DIR)

    print(f"✅ Daily weather parquet written to: {OUT_PARQUET_DIR}")
    spark.stop()

if __name__ == "__main__":
    main()