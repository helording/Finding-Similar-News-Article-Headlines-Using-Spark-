package comp9313.proj3

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{arrays_zip, ceil, col, collect_set, explode, expr, floor, lit, sequence}

object SimilarNews extends App {

  val spark = SparkSession.builder()
    .master("local[1]")
    .appName("SparkByExample")
    .getOrCreate()

  import spark.implicits._

  // Prepare initial Dataframe with id allocated by document order.
  val schema = StructType(Array(StructField("id", IntegerType, nullable = false), StructField("hl", StringType, nullable = false)))
  val file_rdd = spark.sparkContext.textFile(args(0)).zipWithIndex()
  val rdd_row = file_rdd.map(x => Row(x._2.toInt, x._1))
  val initial_df = spark.createDataFrame(rdd_row, schema)

  // Filter for empty headlines
  val data_df = initial_df.filter(functions.length($"hl") =!= 9)

  // Create a Dataframe of headline id's together with their year to use later for filtering. Lazily evaulated until later join.
  val years = data_df.withColumn("years", functions.substring(col("hl"), 0, 4).cast("Int")).drop("hl")
  val min_year = years.agg(functions.min(col("years"))).head().getInt(0)
  val id_year = years.select(col("id"), ($"years" - min_year - lit(128)).cast("Byte").as("year"))

  // Transform the string headlines into a set of integers. Corresponding integer of a word is assigned according to its rank in a word frequency list.
  val set_headlines = data_df.withColumn("hl", functions.split(functions.substring_index(col("hl"), ",", -1), " "))
  val word_freq = set_headlines.select(col("hl").as("word")).withColumn("word", explode(col("word"))).filter(functions.length(col("word")) > 0).groupBy("word").count().sort("count").drop("count")
  val word_freq_rdd = word_freq.select("word").as[String].rdd.zipWithIndex()
  val word_freq_rows: RDD[Row] = word_freq_rdd.map(t => Row(t._1, t._2.toInt))

  val s = new StructType()
    .add(StructField("word", StringType, nullable = false))
    .add(StructField("v", IntegerType, nullable = true))

  val int_words: DataFrame = spark.createDataFrame(word_freq_rows, s)

  val word_list = set_headlines.withColumn("w", explode(Symbol("hl"))).drop("hl").filter(functions.length(col("w")) > 0)
  val int_words_id_map = word_list.join(int_words, $"w" === $"word").select("id", "v")
  val set_hls_as_ints = int_words_id_map.groupBy("id").agg(collect_set(col("v")) as "v_set").drop("v").withColumn("v_set", expr("array_sort(v_set,(l, r) -> case when l < r then -1 when l > r then 1 else 0 end)"))

  // Transform headlines into tokens that can be filtered by prefix and length, resulting in candidate pairs of ids to verify similarity.
  val length_range_added = set_hls_as_ints.withColumn("length", functions.size(col("v_set")).cast("Byte")).withColumn("range", sequence(lit(1), col("length")))
  val hl_sets_range_zipped = length_range_added.withColumn("v_set", arrays_zip(col("range"), col("v_set"))).drop("range")
  val prefix_tokens = hl_sets_range_zipped.withColumn("v", explode(col("v_set"))).drop("v_set").withColumn("index", $"v.range".cast("Byte")).withColumn("v", $"v.v_set") //

  val pruned_tokens = prefix_tokens.filter($"index" <= (col("length") - ceil(col("length") * args(2).toFloat).cast("Int") + 1))

  val pts_year_added = pruned_tokens.as("ex").join(id_year.as("iy"), $"ex.id" === $"iy.id").select("ex.*", "iy.year")

  val length_filtered = pts_year_added.as("df1").join(pts_year_added.as("df2"), $"df1.v" === $"df2.v"
    && floor($"df1.length" * args(2).toFloat) <= $"df2.length"  &&  $"df2.length" <= ceil($"df1.length" / args(2).toFloat)
    && $"df1.id" =!= $"df2.id"
    && $"df1.year" =!= $"df2.year")
    .select("df1.id", "df2.id")

  val candidate_pairs = length_filtered.dropDuplicates().filter(col("df1.id") < col("df2.id"))

  // Join candidate ids with their headlines for similarity calculations. After calculation, filter pairs and format for output.
  val pairs_with__hl_sets = candidate_pairs.join(set_hls_as_ints.as("a1"), $"df1.id" === $"a1.id").join(set_hls_as_ints.as("a2"), $"df2.id" === $"a2.id").select("df1.id", "df2.id", "a1.v_set", "a2.v_set")

  val pair_similarities = pairs_with__hl_sets.withColumn("sim", functions.size(functions.array_intersect(col("a1.v_set"), col("a2.v_set"))).cast("Double") / functions.size(functions.array_union(col("a1.v_set"), col("a2.v_set")))).select("df1.id", "df2.id", "sim").sort("df1.id", "df2.id")

  val filtered_pairs = pair_similarities.filter(col("sim") >= args(2).toDouble)

  val res_rdd = filtered_pairs.rdd.map(x => "(" + x.getInt(0).toString + "," + x.getInt(1).toString + ")" + "\t" + x.getDouble(2).toString)

  res_rdd.saveAsTextFile(args(1))

}