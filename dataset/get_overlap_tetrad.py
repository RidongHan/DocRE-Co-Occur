import os
import ujson as json

dataset = "./docred"
train_set = os.path.join(dataset, "train_annotated.json")
dev_set = os.path.join(dataset, "dev.json")
file_out = os.path.join(dataset, "overlap_triplet_dev.json")

train_set = json.load(open(train_set))
dev_set = json.load(open(dev_set))

overlap_list = []
strict_overlap_list = []
overlap_entity_pair_list = []
max_label_num = 0  # 4

overlap_list_2 = []
strict_overlap_list_2 = []
overlap_entity_pair_list_2 = []
overlap_list_3 = []
strict_overlap_list_3 = []
overlap_entity_pair_list_3 = []
overlap_list_4 = []
strict_overlap_list_4 = []
overlap_entity_pair_list_4 = []

for sample in dev_set:
    title = sample['title']
    labels = sample["labels"]
    tmp_triplet = dict()
    for label in labels:
        if label['h'] == label['t']:  # For dwie dataset
            continue
        r = label['r']
        h = label['h']
        t = label['t']

        if (h, t) not in tmp_triplet:
            tmp_triplet[(h, t)] = [r]
        else:
            tmp_triplet[(h, t)].append(r)

    for k, v in tmp_triplet.items():
        l_num = len(v)
        if l_num > 1:
            if l_num > max_label_num:
                max_label_num = l_num
            if l_num == 2:
                overlap_entity_pair_list_2.append([title, k[0], k[1]])
            elif l_num == 3:
                overlap_entity_pair_list_3.append([title, k[0], k[1]])
            elif l_num == 4:
                overlap_entity_pair_list_4.append([title, k[0], k[1]])
            overlap_entity_pair_list.append([title, k[0], k[1]])
            v.sort()

            strict_overlap_list.append([title, v, k[0], k[1]])
            if l_num == 2:
                strict_overlap_list_2.append([title, v, k[0], k[1]])
            elif l_num == 3:
                strict_overlap_list_3.append([title, v, k[0], k[1]])
            elif l_num == 4:
                strict_overlap_list_4.append([title, v, k[0], k[1]])
            
            for v_1 in v:
                overlap_list.append([title, v_1, k[0], k[1]])
                if l_num == 2:
                    overlap_list_2.append([title, v_1, k[0], k[1]])
                elif l_num == 3:
                    overlap_list_3.append([title, v_1, k[0], k[1]])
                elif l_num == 4:
                    overlap_list_4.append([title, v_1, k[0], k[1]])

print(max_label_num)
print(len(strict_overlap_list), strict_overlap_list, )  # 773
print()
print(len(overlap_list), overlap_list, )  # 1578
print()
print(len(overlap_entity_pair_list), overlap_entity_pair_list, )  # 773

print(len(overlap_list_2), len(overlap_entity_pair_list_2))  # 1490  745
print(len(overlap_list_3), len(overlap_entity_pair_list_3))  # 72    24
print(len(overlap_list_4), len(overlap_entity_pair_list_4))  # 16    4

print(len(strict_overlap_list_2), strict_overlap_list_2)
print()

print(len(strict_overlap_list_3), strict_overlap_list_3)
print()

print(len(strict_overlap_list_4), strict_overlap_list_4)
print()
out_dict = {
    "strict_overlap_list": strict_overlap_list,
    "overlap_list": overlap_list, 
    "overlap_entity_pair_list": overlap_entity_pair_list,
    "strict_overlap_list_2": strict_overlap_list_2, 
    "overlap_list_2": overlap_list_2, 
    "overlap_entity_pair_list_2": overlap_entity_pair_list_2,
    "strict_overlap_list_3": strict_overlap_list_3, 
    "overlap_list_3": overlap_list_3, 
    "overlap_entity_pair_list_3": overlap_entity_pair_list_3,
    "strict_overlap_list_4": strict_overlap_list_4, 
    "overlap_list_4": overlap_list_4, 
    "overlap_entity_pair_list_4": overlap_entity_pair_list_4,
}
json.dump(out_dict, open(file_out, "w"))



##### train set
train_overlap_entity_pair_list = []
train_overlap_list = []

for sample in train_set:
    title = sample['title']
    labels = sample["labels"]
    tmp_triplet = dict()
    for label in labels:
        if label['h'] == label['t']:  # For dwie dataset
            continue
        r = label['r']
        h = label['h']
        t = label['t']

        if (h, t) not in tmp_triplet:
            tmp_triplet[(h, t)] = [r]
        else:
            tmp_triplet[(h, t)].append(r)

    for k, v in tmp_triplet.items():
        l_num = len(v)
        if l_num > 1:
            train_overlap_entity_pair_list.append([title, k[0], k[1]])
            for v_1 in v:
                train_overlap_list.append([title, v_1, k[0], k[1]])
                
print(len(train_overlap_entity_pair_list), len(train_overlap_list))  
        



