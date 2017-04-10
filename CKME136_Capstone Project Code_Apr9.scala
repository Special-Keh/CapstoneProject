import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils


sqlContext.sql("Create database recommender")
sqlContext.sql("use recommender")

%hive
CREATE EXTERNAL TABLE recommender.ratings (
    artist INT,
	track INT,
	userId INT,
	rating DOUBLE,
	time INT
  )
ROW FORMAT DELIMITED
	FIELDS TERMINATED BY ','
	LINES TERMINATED BY '\n'
LOCATION '/user/lab/CapstoneProj/ratings'
tblproperties ("skip.header.line.count"="1")

%hive
LOAD DATA inpath '/user/lab/CapstoneProj/train_clean.csv' OVERWRITE INTO TABLE recommender.ratings


%hive 

CREATE EXTERNAL TABLE recommender.wordsRating (
   	userid2 STRING,
	artist2 STRING,
	avgRating DOUBLE,
	heard_of STRING,
	own_artist STRING,
	like_artist STRING,
	Aggressive INT,
    Edgy INT,
    Thoughtful INT,
    Serious INT,
    GoodLyrics INT,
    Unattractive INT,
    Confident INT,
    Youthful INT,
    Boring INT,
    Current2 INT,
    Cheap INT,
    Calm INT,
    Outgoing INT,
    Inspiring INT,
    Beautiful INT,
    Fun INT,
    Authentic INT,
    Credible INT,
    Cool INT,
    Catchy INT,
    Sensitive INT,
    Superficial INT,
    Passionate INT,
    Timeless INT,
    Depressing INT,
    Original INT,
    Talented INT,
    Distinctive INT,
    Approachable INT,
    Trendsetter INT,
    Noisy INT,
    Upbeat INT,
    Energetic INT,
    None_of_these INT,
    Sexy INT,
    Over2 INT,
    Fake INT,
    Cheesy INT,
    Unoriginal INT,
    Dated INT,
    Unapproachable INT,
    Classic INT,
    Playful INT,
    Arrogant INT,
    Warm INT,
	artistCluster INT
  )
ROW FORMAT DELIMITED
	FIELDS TERMINATED BY ','
	LINES TERMINATED BY '\n'
LOCATION '/user/lab/CapstoneProj/wordsRating'

%hive 
LOAD DATA inpath '/user/lab/CapstoneProj/ArtistDesc_clean.csv' OVERWRITE INTO TABLE recommender.wordsRating


%hive 

CREATE EXTERNAL TABLE recommender.userData (
	userid STRING,
	artist STRING,
	track STRING,
	rating DOUBLE,
	time INT,
	gender STRING,
	age INT,
	workingStatus STRING,
	region STRING,
	music STRING,
	list_own STRING,
	list_back STRING,
	Q1 INT,
	Q2 INT,
	Q3 INT,
	Q4 INT,
	Q5 INT,
	Q6 INT,
	Q7 INT,
	Q8 INT,
	Q9 INT,
	Q10 INT,
	Q11 INT,
	Q12 INT,
	Q13 INT,
	Q14 INT,
	Q15 INT,
	Q16 INT,
	Q17 INT,
	Q18 INT,
	Q19 INT,
	List_Own2 INT,
	List_Back2 INT,
	usercluster INT
  )

ROW FORMAT DELIMITED
	FIELDS TERMINATED BY ','
	LINES TERMINATED BY '\n'
LOCATION '/user/lab/CapstoneProj/userData'

%hive
LOAD DATA inpath '/user/lab/CapstoneProj/trainData_clean.csv' OVERWRITE INTO TABLE recommender.userData


val u = sqlContext.table("userData")
val ad = sqlContext.table("wordsRating")
val r = sqlContext.table("ratings")

// Transform Gender into Numeric Index
val gend_ind = new StringIndexer().
    setInputCol("gender").
    setOutputCol("genderIndex").
    fit(u)

val u2 = gend_ind.transform(u)


// Transform Music into Numeric Index
val music_ind = new StringIndexer().
    setInputCol("music").
    setOutputCol("musicIndex").
    fit(u2)
    
val u3 = music_ind.transform(u2)

// Transform Heard_Of into Numeric Index
val heard_ind = new StringIndexer().
    setInputCol("heard_of").
    setOutputCol("heardIndex").
    fit(ad)
    
val ad2 = heard_ind.transform(ad)

//Save Gender Indexer
gend_ind.save("/user/lab/CapstoneProj/genderIndexer")

//Save Music Indexer
music_ind.save("/user/lab/CapstoneProj/musicIndexer")

//Save Heard Of Indexer
heard_ind.save("/user/lab/CapstoneProj/heardIndexer")


//Create Array of Features variables for Artist Descriptors Table
val ad_features = Array("heardIndex", "aggressive", "edgy", "thoughtful", "serious", "goodlyrics", "unattractive", "confident", "youthful", "boring", "current2", "cheap", "calm", "outgoing", "inspiring", "beautiful", "fun", "authentic", "credible", "cool", "catchy", "sensitive", "superficial", "passionate", "timeless", "depressing", "original", "talented", "distinctive", "approachable", "trendsetter", "noisy", "upbeat", "energetic", "none_of_these", "sexy", "over2", "fake", "cheesy", "unoriginal", "dated", "unapproachable", "classic", "playful", "arrogant", "warm")


//Fill in Null Values
val ad3 = ad2.na.fill(0)

val ad4 = ad3.withColumn("label", ad3("avgRating"))

val ad_vec = new VectorAssembler().
    setInputCols(ad_features).
    setOutputCol("features").
    transform(ad4)

//Split Artist Data Into Training and Test Sets
val Array (ad_train, ad_test) = ad_vec.randomSplit(Array(0.7, 0.3))	
	
//Create Regression Model of Artist Data
val ad_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8).
    fit(ad_train)
   
 println(s"Coefficients: ${ad_glr.coefficients}")
 println(s"Intercept: ${ad_glr.intercept}")
 val trainingSummary = ad_glr.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"MAE = ${trainingSummary.meanAbsoluteError}")
println(s"Explained variance = ${trainingSummary.explainedVariance}")
println(s"r2: ${trainingSummary.r2}")

//Save Artist Descriptors Regression Model
ad_glr.save("/user/lab/CapstoneProj/adRegressionModel")

val adtest_eval = ad_glr.transform(ad_test)
adtest_eval.select($"userid2", $"artist2", $"avgrating",$"label", $"prediction")
val test_MSE = adtest_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = adtest_eval.select($"label", $"prediction").map(v => math.abs(v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double])).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)


//REGRESSION MODEL ON USER DATA
//Create Array of Features variables
val features = Array("genderIndex", "age", "musicIndex", "q1", "q2", "q3", "q4", "q7", "q8", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19")

//Fill in Null Values
val u4 = u3.na.fill(0)

val u5 = u4.withColumn("label", u4("rating"))

val u_vec = new VectorAssembler().
    setInputCols(features).
    setOutputCol("features").
    transform(u5)
	
//Split Data into Training and Test Data Set
val Array (u_train, u_test) = u_vec.randomSplit(Array(0.7, 0.3))

//Create Regression Model
val user_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8)
    
//Fit Model
 val user_model = user_glr.fit(u_train)
 
 println(s"Coefficients: ${user_model.coefficients}")
 println(s"Intercept: ${user_model.intercept}")
 
 val trainingSummary = user_model.summary

println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"MAE = ${trainingSummary.meanAbsoluteError}")
println(s"Explained variance = ${trainingSummary.explainedVariance}")
println(s"r2: ${trainingSummary.r2}")


//Evaluate Model on Test Data
val u_eval = user_model.transform(u_test)

u_eval.select($"userid", $"artist", $"label", $"prediction").show(10)
val test_MSE = u_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = u_eval.select($"label", $"prediction").map(v => math.abs(v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double])).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)



//REGRESSION MODEL ON JOINED DATA SET
// Join Artist and User Ratings Tables
val join_data = u4.join(ad3, u4("userid") === ad3("userid2") && u4("artist") === ad3("artist2"))

//Create Array of Features variables
val features = Array("genderIndex", "musicIndex", "q1", "q2", "q3", "q4", "q7", "q8", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19","heardIndex", "aggressive", "edgy", "thoughtful", "serious", "goodlyrics", "unattractive", "confident", "youthful", "boring", "current2", "cheap", "calm", "outgoing", "inspiring", "beautiful", "fun", "authentic", "credible", "cool", "catchy", "sensitive", "superficial", "passionate", "timeless", "depressing", "original", "talented", "distinctive", "approachable", "trendsetter", "noisy", "upbeat", "energetic", "none_of_these", "sexy", "over2", "fake", "cheesy", "unoriginal", "dated", "unapproachable", "classic", "playful", "arrogant", "warm")

//Fill in Null Values
val join_data2 = join_data.na.fill(0)

//Create Vector of Features
val join_vec = new VectorAssembler().
    setInputCols(features).
    setOutputCol("features").
    transform(join_data2)


/CREATE SEPARATE TABLES FOR RATINGS AND AVERAGE RATINGS
val join_rating = join_vec.select($"userid", $"artist", $"rating".alias("label"), $"features")
val join_avg = join_vec.select($"userid", $"artist", $"avgrating".alias("label"), $"features")

//Model Joined Data On Ratings
//Split Joined Data Set into Training and Test Data
val Array (joinr_train, joinr_test) = join_rating.randomSplit(Array(0.7, 0.3))

//Create Regression Model for Joined Dataset
val joinr_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8).
    fit(joinr_train)
 
 println(s"Coefficients: ${joinr_glr.coefficients}")
 println(s"Intercept: ${joinr_glr.intercept}")

val trainingSummary = joinr_glr.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"MAE = ${trainingSummary.meanAbsoluteError}")
println(s"Explained variance = ${trainingSummary.explainedVariance}")
println(s"r2: ${trainingSummary.r2}")

//Evaluate Model on Test Data
val joinrtest_eval = joinr_glr.transform(joinr_test)

val test_MSE = joinrtest_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = joinrtest_eval.select($"label", $"prediction").map(v => math.abs(v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double])).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)

//Save JoinedData Rating Regression Model
joinr_glr.save("/user/lab/CapstoneProj/joinRatingRegModel")



//Model Joined Data On Average Ratings
//Split Joined Data Set into Training and Test Data
val Array (joina_train, joina_test) = join_avg.randomSplit(Array(0.7, 0.3))

//Create Regression Model for Joined Dataset
val joina_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8).
    fit(joina_train)
 
 println(s"Coefficients: ${joina_glr.coefficients}")
 println(s"Intercept: ${joina_glr.intercept}")

val trainingSummary = joina_glr.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"MAE = ${trainingSummary.meanAbsoluteError}")
println(s"Explained variance = ${trainingSummary.explainedVariance}")
println(s"r2: ${trainingSummary.r2}")

//Evaluate Model on Test Data
val joinatest_eval = joina_glr.transform(joina_test)
val test_MSE = joinatest_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = joinatest_eval.select($"label", $"prediction").map(v => math.abs(v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double])).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)

//Save JoinedData Average Rating Regression Model
joina_glr.save("/user/lab/CapstoneProj/joinedAvgRegModel")

val globalAvg = r.agg(avg("rating").alias("avgRating"))
globalAvg.show()
//Calcualte Error Metrics for predictions of Global Average Ratings
val test_MSE = join_avg.select($"label").map(v => math.pow((v(0).asInstanceOf[Double] - 36.44), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = join_avg.select($"label").map(v => math.abs(v(0).asInstanceOf[Double] - 36.44)).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)

//Calcualte Error Metrics for predictions of Global Average Ratings
val test_MSE = r.select($"rating").map(v => math.pow((v(0).asInstanceOf[Double] - 36.44), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
val test_MAE = r.select($"rating").map(v => math.abs(v(0).asInstanceOf[Double] - 36.44)).mean()
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
println("test Mean Absolute Error = " + test_MAE)


//NAIVEBAYES CLASSIFICATION
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint


//NAIVE BAYES CLASSIFICATION OF AVERAGE RATINGS - Joined Data Set 4 Categories
val naivejoin_avg = join_vec.select($"userid", $"artist", $"avgrating", $"features")

val splits = Array(0.0, 25.0, 50.0, 75.0, 100.0)
val bucketed_data = new Bucketizer().
    setInputCol("avgrating").
    setOutputCol("label").
    setSplits(splits).
    transform(naivejoin_avg)
    
val Array (joina_trainNaive, joina_testNaive) = bucketed_data.randomSplit(Array(0.7, 0.3))

val bucketed_train = joina_trainNaive.select($"userid", $"artist", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

val bucketed_test = joina_testNaive.select($"userid", $"artist", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
   
val nb_model = NaiveBayes.train(bucketed_train, lambda = 1.0, modelType = "multinomial")

val prediction_label = bucketed_test.map(p=> (nb_model.predict(p.features), p.label))
val accuracy = 1.0 * prediction_label.filter(x => x._1 == x._2).count() / bucketed_test.count()

// Instantiate metrics object
val metrics = new MulticlassMetrics(prediction_label)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

//Distribution of Ratings in Data Set
joina_trainNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/joina_trainNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
joina_testNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/joina_testNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
bucketed_data.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/bucketed_data.count())*100).alias("Percent")).orderBy($"label".asc).show()


//NAIVE BAYES CLASSIFICATION OF AVG RATINGS USING ARTIST DESCRIPTORS ONLY 4 Categories
val naive_ad = ad_vec.select($"userid2", $"artist2", $"avgrating", $"features")

val splits = Array(0.0, 25.0, 50.0, 75.0, 100.0)
val bucketed_data3 = new Bucketizer().
    setInputCol("avgrating").
    setOutputCol("label").
    setSplits(splits).
    transform(naive_ad)
    
val Array (ad_trainNaive, ad_testNaive) = bucketed_data3.randomSplit(Array(0.7, 0.3))

val bucketed_adtrain = ad_trainNaive.select($"userid2", $"artist2", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

val bucketed_adtest = ad_testNaive.select($"userid2", $"artist2", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
   
val nb_model3 = NaiveBayes.train(bucketed_adtrain, lambda = 1.0, modelType = "multinomial")

val prediction_label3 = bucketed_adtest.map(p=> (nb_model3.predict(p.features), p.label))
val accuracy = 1.0 * prediction_label3.filter(x => x._1 == x._2).count() / bucketed_adtest.count()

// Instantiate metrics object
val metrics = new MulticlassMetrics(prediction_label3)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

//Distribution of Ratings in Data Set
ad_trainNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/ad_trainNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
ad_testNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/ad_testNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
bucketed_data3.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/bucketed_data3.count())*100).alias("Percent")).orderBy($"label".asc).show()


//NAIVE BAYES CLASSIFICATION OF AVERAGE RATINGS JOINED DATA - 5 Categories
val naivejoin_avg = join_vec.select($"userid", $"artist", $"avgrating", $"features")

val splits2 = Array(0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
val bucketed_data = new Bucketizer().
    setInputCol("avgrating").
    setOutputCol("label").
    setSplits(splits2).
    transform(naivejoin_avg)
    
val Array (joina_trainNaive, joina_testNaive) = bucketed_data.randomSplit(Array(0.7, 0.3))

val bucketed_train = joina_trainNaive.select($"userid", $"artist", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

val bucketed_test = joina_testNaive.select($"userid", $"artist", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
   
val nb_model = NaiveBayes.train(bucketed_train, lambda = 1.0, modelType = "multinomial")

val prediction_label = bucketed_test.map(p=> (nb_model.predict(p.features), p.label))
val accuracy = 1.0 * prediction_label.filter(x => x._1 == x._2).count() / bucketed_test.count()

// Instantiate metrics object
val metrics = new MulticlassMetrics(prediction_label)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


//Distribution of Ratings in Data Set
joina_testNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/joina_testNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
joina_trainNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/joina_trainNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
bucketed_data.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/bucketed_data.count())*100).alias("Percent")).orderBy($"label".asc).show()


//NAIVE BAYES CLASSIFICATION OF RATINGS USING ARTIST DESCRIPTORS ONLY - 5 Categories
val naive_ad = ad_vec.select($"userid2", $"artist2", $"avgrating", $"features")

val splits = Array(0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
val bucketed_data5 = new Bucketizer().
    setInputCol("avgrating").
    setOutputCol("label").
    setSplits(splits).
    transform(naive_ad)
    
val Array (ad_trainNaive, ad_testNaive) = bucketed_data5.randomSplit(Array(0.7, 0.3))

val bucketed_adtrain = ad_trainNaive.select($"userid2", $"artist2", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))

val bucketed_adtest = ad_testNaive.select($"userid2", $"artist2", $"avgrating", $"features", $"label").rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
   
val nb_model5 = NaiveBayes.train(bucketed_adtrain, lambda = 1.0, modelType = "multinomial")

val prediction_label5 = bucketed_adtest.map(p=> (nb_model5.predict(p.features), p.label))
val accuracy = 1.0 * prediction_label5.filter(x => x._1 == x._2).count() / bucketed_adtest.count()

// Instantiate metrics object
val metrics = new MulticlassMetrics(prediction_label5)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

//Distribution of Ratings in Data Set
ad_trainNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/ad_trainNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
ad_testNaive.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/ad_testNaive.count())*100).alias("Percent")).orderBy($"label".asc).show()
bucketed_data5.groupBy($"label").agg(count($"label").alias("Count"),((count($"label")/bucketed_data5.count())*100).alias("Percent")).orderBy($"label".asc).show()




//SELECT USER & PROVIDE RECOMMENDATIONS
//Select Rated Artists
sqlContext.sql("use recommender")
val u = sqlContext.table("userData")
val ad = sqlContext.table("wordsRating")
val r = sqlContext.table("ratings")

val ratedArtists = r.select($"*").filter($"userId" === selUser)

val l = ratedArtists.select($"artist").distinct

val unRated = ad4.join(l, ad4("artist2") !== l("artist"))

//Select Users Profile
val userProfile = u_vec.select($"userid", $"gender", $"age", $"workingStatus", $"region", $"music", $"Q1", $"Q2", $"Q3", $"Q4", $"Q5", $"Q6", $"Q7", $"Q8", $"Q9", $"Q10", $"Q11", $"Q12", $"Q13", $"Q14", $"Q15", $"Q16", $"Q17", $"Q18", $"Q19", $"List_Own2", $"List_Back2", $"genderIndex", $"musicIndex", $"usercluster").filter($"userid" === selUser).limit(1)

val recommendations = userProfile.join(unRated)

//Predict Artist Ratings
//Create Array of Features variables
val features = Array("genderIndex", "musicIndex", "Q1", "Q2", "Q3", "Q4", "Q7", "Q8", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19","heardIndex", "aggressive", "edgy", "thoughtful", "serious", "goodlyrics", "unattractive", "confident", "youthful", "boring", "current2", "cheap", "calm", "outgoing", "inspiring", "beautiful", "fun", "authentic", "credible", "cool", "catchy", "sensitive", "superficial", "passionate", "timeless", "depressing", "original", "talented", "distinctive", "approachable", "trendsetter", "noisy", "upbeat", "energetic", "none_of_these", "sexy", "over2", "fake", "cheesy", "unoriginal", "dated", "unapproachable", "classic", "playful", "arrogant", "warm")

//Create Vector of Features
val recommendations_vec = new VectorAssembler().
    setInputCols(features).
    setOutputCol("features").
    transform(recommendations)
    
//Load Model
val recoModel = LinearRegressionModel.load("/user/lab/CapstoneProj/joinedAvgRegModel")

val predictedRatings = recoModel.transform(recommendations_vec)
val usersRecommendations = predictedRatings.select($"artist2", $"prediction").groupBy($"artist2").agg(avg($"prediction").alias("avgRating")).orderBy($"avgRating".desc)

val predictedRatings = recoModel.transform(recommendations_vec)
val usersRecommendations = predictedRatings.select($"artist2", $"prediction").groupBy($"artist2").agg(avg($"prediction").alias("avgRating")).orderBy($"avgRating".desc)

println("Hello " + selUser + "! Here are some other Artists you might like:")
usersRecommendations.select($"artist2").show(10)
