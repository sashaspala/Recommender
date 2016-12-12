
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext("local", "Recommendation")
data_file = sc.textFile("file:///home/hadoop02/ratings-small-hashes.txt")
ratings = data_file.map(lambda l: l.split(','))\
   .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))


# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

features_matrix = model.productFeatures()
features_dict = {}

# create dictionary where key is hashed item id, value is feature vector
for feature_element in features_matrix.toLocalIterator():
    features_dict[feature_element[0]] = feature_element[1]

# create dictionary of hashed item id, Amazon item id pairs
id_dict = {}
id_file = open("id_dict-medium.txt").read()
id_file = id_file.splitlines()
for line in id_file:
    ids = line.split(',')
    id_dict[ids[0]]=ids[1]

# read in Amazon ids to find most similar items of
item_file = open("items.txt").read()
item_file = item_file.splitlines()


# create list where first element is hashed item id, second is similarity
recommended_products = [(None, 0) * 10]
lowest_similarity = 0


for line in item_file:
    item_ids = line.split(',')


    item = item_ids[0]
    original_id = item_ids[1]

    id_dict[item] = original_id

    if features_dict.get(item) is None:
        continue

    current_feature_vector = features_dict.get(item)

    for feature in features_matrix:

        if item != feature[0]:

            compare_to_vector = feature[1]

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

    for hashed_id in recommended_products:
        print(items_dict[hashed_id])

    recommended_products = [(None, 0) * 10]
