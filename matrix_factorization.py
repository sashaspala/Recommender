
import sys
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Load and parse the data

sc = SparkContext("local", "Recommendation")
data_file = sc.TextFile("ratings-small.txt")
ratings = data_file.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# I don't think the two following two blocks of code are necessary
# Evaluate the model on training data
# testdata = ratings.map(lambda p: (p[0], p[1]))
# predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# print("Mean Squared Error = " + str(MSE))

# Save and load model
# model.save(data_file, "target/tmp/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(data_file, "target/tmp/myCollaborativeFilter")

features_matrix = model.productFeatures()
features_dict = {}

### put into dictionary for quicker retrieval
for feature_element in features_matrix:
    features_dict[feature_element[0]] = feature_element[1]

item_file = sc.TextFile("items.txt").read()
item_file = item_file.splitlines()

recommended_products = [(None, 0) * 10]
lowest_similarity = 0

for item in item_file:
    if features_dict.get(item) == None:
        continue
    current_feature_vector = features_dict.get(item)
    for feature in features_matrix:
        if item != feature[0]:
            compare_to_vector = feature[1]
            result_array = []
            for index in range(len(compare_to_vector)):
                result_array[index] = compare_to_vector[index] * current_feature_vector[index]
            sum = 0
            for value in result_array:
                sum = sum + value

            if sum > lowest_similarity:
                recommended_products.append((sum, feature[0]))

                for element in recommended_products:
                    if element[1] < feature[0]:
                        ##get the lowest value
                        temp_similarity = element[1]
                    if element[1] == lowest_similarity:
                        #find my last lowest similarity and remove it
                        recommended_products.remove(element)
                        break

                #now reset to the right lowest similarity
                lowest_similarity = temp_similarity

    print(recommended_products)
    recommended_products = [(None, 0) * 10]