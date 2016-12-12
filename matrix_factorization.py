
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

print(model)

features_matrix = model.productFeatures()
features_dict = {}

# create dictionary where key is hashed item id, value is feature vector
for feature_element in features_matrix.toLocalIterator():
    features_dict[feature_element[0]] = feature_element[1]

# create 2 dictionaries
# 1st dict is key hashed item id, value Amazon item id
hash_to_amazon = {}

# 2nd dict is key amazon id, value hashed id
amazon_to_hash = {}

id_file = open("id_dict-small.txt").read()
id_file = id_file.splitlines()
for line in id_file:
    ids = line.split(',')
    hash_to_amazon[ids[0]]=ids[1]
    amazon_to_hash[ids[1]] = ids[0]

# read in Amazon ids to find most similar items of
item_file = open("items.txt").read()
amazon_items = item_file.splitlines()


# create empty list where first element is hashed item id, second is similarity
recommended_products = [(None, 0) * 10]
lowest_similarity = 0


for item in amazon_items:
    hashed_item = amazon_to_hash[item]

    if features_dict.get(hashed_item) is None:
        continue

    current_feature_vector = features_dict.get(hashed_item)

    for feature in features_matrix:
        if hashed_item != feature[0]:
            compare_to_vector = feature[1]
            dot_product = sum([i * j for (i, j) in zip(current_feature_vector, compare_to_vector)])
            if dot_product > lowest_similarity:
                recommended_products.append((feature[0], dot_product))
                recommended_products = sorted(recommended_products, key=lambda x: x[1])
                recommended_products = recommended_products[10:]

    for hashed_item in recommended_products:
        print(hash_to_amazon[hashed_item])

    recommended_products = [(None, 0) * 10]
