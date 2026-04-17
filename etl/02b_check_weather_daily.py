from pyspark.sql import SparkSession
from pyspark.sql.functions import min as spark_min, max as spark_max, countDistinct

spark = SparkSession.builder.appName("CheckWeatherDaily").getOrCreate()

w = spark.read.parquet("Data/processed/parquet/weather_daily")

w.printSchema()

w.select(
    spark_min("date").alias("min_date"),
    spark_max("date").alias("max_date"),
    countDistinct("city_norm").alias("num_cities")
).show(truncate=False)

w.groupBy("city_norm").count().orderBy("city_norm").show(50, truncate=False)

spark.stop()