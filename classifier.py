from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

spark = SparkSession.builder.appName("Classifier").master("local[*]").getOrCreate()

if __name__ == "__main__":
    def mapper1(r,v):
        if v == 1:
            v1 = "T3"; v2 = "T4"
        elif v ==2:
            v1 = "T3"; v2 = "Tamb"
        else:
            v1 = "Oil"; v2 = "FS2"
        splited = r[0].split(' ')
        if (splited[0] == v1 or splited[0] == v2):
            return ((int(splited[1]),int(splited[2])),float(splited[3]))

    tMapper = spark.read.text("CE-CP.txt").rdd.map(lambda r: mapper1(r,1))\
                   .filter(lambda x: x is not None).reduceByKey(lambda x,y: x-y)

    tMapper2 = spark.read.text("CE-CP.txt").rdd.map(lambda r: mapper1(r,2))\
                    .filter(lambda x: x is not None).reduceByKey(lambda x,y: x-y)
    Q1 = spark.read.text("CE-CP.txt").rdd.map(lambda r: mapper1(r,3))\
              .filter(lambda x: x is not None).reduceByKey(lambda x,y: (x*y)/(6*10**7))

    CE = tMapper.union(tMapper2).reduceByKey(lambda x,y: (x/y)*100).sortByKey()\
                .map(lambda r: (r[0][0],r[1])).groupByKey().map(lambda r: (r[0],list(r[1])))
    CP = tMapper.union(Q1).reduceByKey(lambda x,y: x*y).sortByKey()\
                .map(lambda r: (r[0][0],r[1])).groupByKey().map(lambda r: (r[0],list(r[1])))

def mapper(r):
    t =[]
    splited = r[0].split(' ')
    for i in range(1,len(splited)):
        t.append(float(splited[i]))
    return(int(splited[0]),t)
    
def LabelMapper(r):
    t =[]
    splited = r[1][0]
    for i in range(0,len(splited)):
        t.append(float(splited[i]))
    t.append(r[1][1])
    return(r[0],t,len(t))
    
def UnionMapper(r):
    t =[]
    splited = r[1][0]
    for i in range(0,len(splited)):
        t.append(float(splited[i]))
    splited = r[1][1]
    for i in range(0,len(splited)):
        t.append(float(splited[i]))
    return(r[0],t)

SE = spark.read.text("NewSE.txt").rdd.map(lambda r: mapper(r))
PS1 = spark.read.text("NewPS1.txt").rdd.map(lambda r: mapper(r))
PS3 = spark.read.text("NewPS3.txt").rdd.map(lambda r: mapper(r))

labels = spark.read.text("NewProfile.txt").rdd.map(lambda r: mapper(r))

CE_CP = CE.union(CP).reduceByKey(lambda x,y: (x,y)).map(lambda r : UnionMapper(r))
CE_CP_SE = CE_CP.union(SE).reduceByKey(lambda x,y: (x,y)).map(lambda r : UnionMapper(r))
CE_CP_SE_PS1 = CE_CP_SE.union(PS1).reduceByKey(lambda x,y: (x,y)).map(lambda r : UnionMapper(r))
CE_CP_SE_PS1_PS3 = CE_CP_SE_PS1.union(PS3).reduceByKey(lambda x,y: (x,y)).map(lambda r : UnionMapper(r))

TotalData = CE_CP_SE_PS1_PS3.union(labels).reduceByKey(lambda x,y: (x,y[4])).map(lambda r : LabelMapper(r))

transformed_df = TotalData.map(lambda row: LabeledPoint(row[1][-1], Vectors.dense(row[1][0:-1])))

print("Total number of rows: %d" % transformed_df.count())

splits = [0.8, 0.2]
training_data, test_data = transformed_df.randomSplit(splits, 13579)


print("Number of training set rows: %d" % training_data.count())
print("Number of test set rows: %d" % test_data.count())
#
model = RandomForest.trainClassifier(training_data, numClasses = 2, categoricalFeaturesInfo={}, \
    numTrees=3, featureSubsetStrategy="auto", impurity="gini", \
    maxDepth=4, maxBins=32, seed=13579)

predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Model accuracy: %.3f%%" % (acc * 100))
