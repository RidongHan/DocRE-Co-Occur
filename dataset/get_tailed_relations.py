import os
import ujson as json
import numpy as np

dataset = "./docred"
dev_set = os.path.join(dataset, "train_annotated.json")
rel2id = os.path.join(dataset, "rel2id.json")
file_out = os.path.join(dataset, "rel_500_200_100.json")
num_class = 97

rel2id = json.load(open(rel2id))
id2rel = {v:k for k, v in rel2id.items()}
dev_set = json.load(open(dev_set))

num_relation_fact = [0] * num_class

for sample in dev_set:
    labels = sample["labels"]
    for label in labels:
        rel = label["r"]
        rel_id = rel2id[rel]
        num_relation_fact[rel_id] += 1

print(sum(num_relation_fact), num_relation_fact)
array_b = np.array(num_relation_fact)
print(np.argsort(-array_b))

rel_rich = []
rel_rich_1 = []
rel_500 = []
rel_200 = []
rel_100 = []
for i in range(1, num_class):
    if num_relation_fact[i] < 100:
        rel_100.append(i)
        rel_200.append(i)
        rel_500.append(i)
    elif num_relation_fact[i] < 200:
        rel_200.append(i)
        rel_500.append(i)
    elif num_relation_fact[i] < 500:
        rel_500.append(i)
        rel_rich_1.append(i)
    else:
        rel_rich.append(i)
        rel_rich_1.append(i)
print(len(rel_rich_1), rel_rich_1)
print()
print(len(rel_rich), rel_rich)
print()
print(len(rel_500), rel_500)
print()
print(len(rel_200), rel_200)
print()
print(len(rel_100), rel_100)

rel_list = {}

list_500 = {}
list_200 = {}
list_100 = {}
list_all = {}

for i in range(1, len(num_relation_fact)):
    rel = id2rel[i]
    num = num_relation_fact[i]
    list_all[rel] = i
    if num<100:
        list_500[rel] = i
        list_200[rel] = i
        list_100[rel] = i
    elif num<200:
        list_500[rel] = i
        list_200[rel] = i
    elif num<500:
        list_500[rel] = i

rel_list = {"rall":list_all, "r500": list_500, "r200": list_200, "r100": list_100}
print(rel_list)
print(len(list_all), len(list_500), len(list_200), len(list_100))
json.dump(rel_list, open(file_out, "w"))



