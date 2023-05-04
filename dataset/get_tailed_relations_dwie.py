import os
import ujson as json
import numpy as np

dataset = "./dwie"
dev_set = os.path.join(dataset, "train_annotated.json")
rel2id = os.path.join(dataset, "rel2id.json")
file_out = os.path.join(dataset, "rel_tailed.json")
num_class = 66

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

rel_list = {}

list_200 = {}
list_150 = {}
list_100 = {}
list_50 = {}
list_20 = {}
list_10 = {}
list_all = {}

for i in range(1, len(num_relation_fact)):
    rel = id2rel[i]
    num = num_relation_fact[i]
    list_all[rel] = i
    if num<10:
        list_200[rel] = i
        list_150[rel] = i
        list_100[rel] = i
        list_50[rel] = i
        list_20[rel] = i
        list_10[rel] = i
    elif num<20:
        list_200[rel] = i
        list_150[rel] = i
        list_100[rel] = i
        list_50[rel] = i
        list_20[rel] = i
    elif num<50:
        list_200[rel] = i
        list_150[rel] = i
        list_100[rel] = i
        list_50[rel] = i
    elif num<100:
        list_200[rel] = i
        list_150[rel] = i
        list_100[rel] = i
    elif num<150:
        list_200[rel] = i
        list_150[rel] = i
    elif num<200:
        list_200[rel] = i

rel_list = {"rall":list_all, "r200": list_200, "r150": list_150, "r100": list_100, "r50": list_50, "r20": list_20, "r10": list_10}
print(rel_list)
for k, v in rel_list.items():
    print(k, len(v))
json.dump(rel_list, open(file_out, "w"))



