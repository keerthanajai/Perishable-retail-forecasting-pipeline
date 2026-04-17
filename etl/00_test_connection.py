from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("PostgresTest") \
    .getOrCreate()

data = [("Spark", 1)]
df = spark.createDataFrame(data, ["name", "value"])

jdbc_url = "jdbc:postgresql://localhost:5432/favorita"

connection_properties = {
    "user": "postgres",
    "password": "Jaikee@27",
    "driver": "org.postgresql.Driver"
}

df.write.jdbc(
    url=jdbc_url,
    table="retail.test_table",
    mode="overwrite",
    properties=connection_properties
)

spark.stop()