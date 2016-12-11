
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from collections import namedtuple
import pickle
'''
class Rating(namedtuple("Rating", ["user", "product", "rating"])):
    """Takes a string userID, string productID, and int rating, and returns Rating obj
    like the Rating obj in pyspark.mllib.recommendation, with userID and productID stored as strings not ints"""

    def __init__(self, user, product, rating):
        self.user = user
        self.product = product
        self.rating = rating

    def __reduce__(self):
        return Rating, (str(self.user), str(self.product), float(self.rating))
# Load and parse the data
'''

sc = SparkContext("local", "Recommendation")
data_file = sc.textFile("file:///home/hadoop02/ratings-small-no-hapaxes.txt")
ratings = data_file.map(lambda l: l.split(','))\
    .map(lambda l: Rating(binascii.b2a_hex(pickle.dump(str(l[0]))), binascii.b2a_hex(pickle.dump(str(l[1]))), float(l[2])))

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

# create list where first element is item id, second is similarity
recommended_products = [(None, 0) * 10]
lowest_similarity = 0

# we're assuming that the elements in the features matrix are
# indexed under a name in the same format as how they're written in the items.txt file

for item in item_file:

    if features_dict.get(item) is None:
        continue

    current_feature_vector = features_dict.get(item)

    for feature in features_matrix:

        if item != feature[0]:

            compare_to_vector = feature[1]

            # result_array = []
            # for index in range(len(compare_to_vector)):
            #    result_array[index] = compare_to_vector[index] * current_feature_vector[index]
            # sum = 0
            # for value in result_array:
            #     sum = sum + value

            current_sum = sum([i * j for (i, j) in zip(current_feature_vector, compare_to_vector)])

            if current_sum > lowest_similarity:

                # first element in tuple is item id, second is similarity
                recommended_products.append((feature[0], current_sum))

                for element in recommended_products:

                    # isn't this comparing similarity to an item id?
                    # if element[1] < feature[0]:

                    if element[1] < current_sum:
                        ##get the lowest value
                        temp_similarity = element[1]

                    if element[1] == lowest_similarity:
                        #find my last lowest similarity and remove it
                        recommended_products.remove(element)
                        break

                #now reset to the right lowest similarity
                lowest_similarity = temp_similarity

    print(pickle.load(recommended_products))
    recommended_products = [(None, 0) * 10]
