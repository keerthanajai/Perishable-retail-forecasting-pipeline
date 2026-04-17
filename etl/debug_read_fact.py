import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.sql import SparkSession
from config.db_config import JDBC_URL, DB_PROPERTIES

spark = (
    SparkSession.builder
    .appName("DebugReadFact")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

fact = spark.read.jdbc(
    url=JDBC_URL,
    table="retail.fact_sales",
    column="store_id",
    lowerBound=1,
    upperBound=54,
    numPartitions=8,
    properties=DB_PROPERTIES
)

print("Sample rows:")
fact.show(5)

print("Distinct store_ids:", fact.select("store_id").distinct().count())

spark.stop()