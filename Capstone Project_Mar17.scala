
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.ml.regression.LinearRegressionModel


sqlContext.sql("Create database recommender")

sqlContext.sql("use recommender")

%hive
CREATE EXTERNAL TABLE recommender.ratings (
    artist STRING,
	track STRING,
	userId STRING,
	rating INT,
	time INT
  )
ROW FORMAT DELIMITED
	FIELDS TERMINATED BY ','
	LINES TERMINATED BY '\n'
LOCATION '/user/lab/CapstoneProj/ratings'

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



//REGERSSION MODEL ON ARTIST DATA
// Transform Heard_Of into Numeric Index
val heard_ind = new StringIndexer().
    setInputCol("heard_of").
    setOutputCol("heardIndex").
    fit(ad)
    
val ad2 = heard_ind.transform(ad)
//Save Heard Of Indexer
gend_ind.save("/user/lab/CapstoneProj/heardIndexer")


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

 
//Summary Statistics of Regression Model
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

//Test Model on Test Dataset
val test_eval = ad_glr.transform(ad_test)

val test_MSE = test_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)




//USER DATA REGRESSION MODEL
// Modelling of User Data
// Transform Gender into Numeric Index
val gend_ind = new StringIndexer().
    setInputCol("gender").
    setOutputCol("genderIndex").
    fit(u)

val u2 = gend_ind.transform(u)
//Save Gender Indexer
gend_ind.save("/user/lab/CapstoneProj/genderIndexer")

// Transform Music into Numeric Index
val music_ind = new StringIndexer().
    setInputCol("music").
    setOutputCol("musicIndex").
    fit(u2)
    
val u3 = music_ind.transform(u2)
//Save Music Indexer
music_ind.save("/user/lab/CapstoneProj/musicIndexer")


//Create Array of Features variables
val features = Array("genderIndex", "age", "musicIndex", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "list_own2", "list_back2")

//Fill in Null Values
val u4 = u3.na.fill(0)

val u5 = u4.withColumn("label", u4("rating"))

val u_vec = new VectorAssembler().
    setInputCols(features).
    setOutputCol("features").
    transform(u5)
	
//Split Data into Training and Test Data Set
val Array (train_data, test_data) = u_vec.randomSplit(Array(0.7, 0.3))

//Create Regression Model
val u_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8)
    
//Fit Model
 val u_model = u_glr.fit(train_data)
 
 println(s"Coefficients: ${model.coefficients}")
 println(s"Intercept: ${model.intercept}")
 
 val trainingSummary = u_model.summary

println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

//Save User Regression Model
u_glr.save("/user/lab/CapstoneProj/userRegressionModel")

//Evaluate Model on Test Data
val u_eval = user_model.transform(u_test)

u_eval.select($"userid", $"artist", $"label", $"prediction").show(10)
val test_MSE = u_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)



//CREATE MODEL WITH COMBINED DATASET
// Join Artist and User Ratings Tables
val join_data = u4.join(ad3, u4("userid") === ad3("userid2") && u4("artist") === ad3("artist2"))


//Create Array of Features variables
val features = Array("genderIndex", "age", "musicIndex", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "list_own2", "list_back2", "heardIndex", "aggressive", "edgy", "thoughtful", "serious", "goodlyrics", "unattractive", "confident", "youthful", "boring", "current2", "cheap", "calm", "outgoing", "inspiring", "beautiful", "fun", "authentic", "credible", "cool", "catchy", "sensitive", "superficial", "passionate", "timeless", "depressing", "original", "talented", "distinctive", "approachable", "trendsetter", "noisy", "upbeat", "energetic", "none_of_these", "sexy", "over2", "fake", "cheesy", "unoriginal", "dated", "unapproachable", "classic", "playful", "arrogant", "warm")


//Fill in Null Values
val join_data2 = join_data.na.fill(0)

//Create Vector of Features
val join_vec = new VectorAssembler().
    setInputCols(features).
    setOutputCol("features").
    transform(join_data2)

val join_rating = join_vec.select($"userid", $"artist", $"rating".alias("label"), $"features")
val join_avg = join_vec.select($"userid", $"artist", $"avgrating".alias("label"), $"features")
	
//MODEL JOINED DATA ON RATINGS
//Split Joined Data Set into Training and Test Data
val Array (joinr_train, joinr_test) = join_rating.randomSplit(Array(0.7, 0.3))

//Create Regression Model for Joined Dataset
val joinr_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8).
    fit(joinr_train)
 
 println(s"Coefficients: ${model.coefficients}")
 println(s"Intercept: ${model.intercept}")

val trainingSummary = rating_model.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

//Save JoinedData Rating Regression Model
joinr_glr.save("/user/lab/CapstoneProj/joinRatingRegModel")

//Evaluate Model on Test Data
val joinrtest_eval = joinr_glr.transform(joinr_test)

joinrtest_eval.select($"userid", $"artist", $"label", $"prediction").show(10)
val test_MSE = test_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)



//MODEL JOINED DATA ON AVERAGE RATINGS
//Split Joined Data Set into Training and Test Data
val Array (joina_train, joina_test) = join_avg.randomSplit(Array(0.7, 0.3))

//Create Regression Model for Joined Dataset
val joina_glr = new LinearRegression().
    setMaxIter(10).
    setRegParam(0.3).
    setElasticNetParam(0.8).
    fit(joina_train)
 
 println(s"Coefficients: ${model.coefficients}")
 println(s"Intercept: ${model.intercept}")

val trainingSummary = joina_glr.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

//Save JoinedDate Average Rating Regression Model
joina_glr.save("/user/lab/CapstoneProj/joinedAvgRegModel")

//Evaluate Model on Test Data
val joinatest_eval = joina_glr.transform(joina_test)
joinatest_eval.select($"userid", $"artist", $"label", $"prediction").show(10)
val test_MSE = joinatest_eval.select($"label", $"prediction").map(v => math.pow((v(0).asInstanceOf[Double] - v(1).asInstanceOf[Double]), 2)).mean()
val test_RMSE = math.sqrt(test_MSE)
println("test Mean Squared Error = " + test_MSE)
println("test Root Mean Squared Error = " + test_RMSE)
