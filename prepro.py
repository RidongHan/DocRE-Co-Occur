from tqdm import tqdm
import ujson as json
import numpy as np
from collections import defaultdict
import os


def read_docred(args, file_in, file_out, tokenizer, docred_rel2id, max_seq_length=1024, logger=None):
    if os.path.exists(file_out):
        with open(file_out, "r") as fh:
            features = json.load(fh)
            logger.write(file_out + " has been loaded! | # of the document: " + str(len(features)) + "\n")
        return features

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    self_relation_num = 0
    single_or_no_entity_num = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    if args.joint_label_embed:  # label token ids
        rel_list = []
        for i in range(args.num_class):
            rel_list.append("[rel-" + str(i) + "]")
        rel_list.append(tokenizer.sep_token)
        rel_seqs = " ".join(rel_list)

        rel_seqs_tokens = tokenizer.tokenize(rel_seqs)
        rel_input_ids = tokenizer.convert_tokens_to_ids(rel_seqs_tokens)
        token_length = max_seq_length - len(rel_input_ids)
    else:
        token_length = max_seq_length


    for sample in tqdm(data, desc="Example"):
        mentions_info = []  # for graph  # start | end | entityid | sentid
        entities_info = []  # for graph
        sentences_info = []  # for graph

        ###### for sentences_info
        Ls = [0]  
        for sent in sample['sents']:
            Ls.append(len(sent))  
        ###### for sentences_info

        sents = []
        sent_map = []

        entities = sample['vertexSet']
        if len(entities) < 2:  # For dwie dataset
            single_or_no_entity_num += 1
            # print(sample)
            continue
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        ###### for sentences_info
        new_Ls = [0]  
        for idx in range(1, len(Ls)):
            new_Ls.append(sent_map[idx-1][Ls[idx]])
            if new_Ls[idx-1] < token_length:
                sentences_info.append([new_Ls[idx-1], new_Ls[idx], -1, idx-1])  
        ###### for sentences_info

        train_triple = {}
        relation_set = []
        if "labels" in sample:
            for label in sample['labels']:
                if label['h'] == label['t']:  # For dwie dataset
                    self_relation_num += 1
                    continue
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if r not in relation_set:
                    relation_set.append(r)
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e_id, e in enumerate(entities):
            entities_info.append([-1, -1, e_id, -1])
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                if start < token_length:
                    mentions_info.append([start, end, e_id, m["sent_id"]])

        relations, hts = [], []

        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:token_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        sent_input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        if args.joint_label_embed:
            input_ids = sent_input_ids + rel_input_ids
        else:
            input_ids = sent_input_ids

        labels_mask = [1.0] * len(train_triple) + [0.0] * (len(entities) * (len(entities)-1) - len(train_triple))
        assert len(labels_mask) == len(entities) * (len(entities)-1), "Error labels_mask"
        i_line += 1

        nodes_adj = create_graph(mentions_info, entities_info, sentences_info)

        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'labels_mask': labels_mask,
                   'title': sample['title'],
                   'label_set': relation_set, 
                   'label_start_ids': len(sent_input_ids),
                   'mentions_info': mentions_info,
                   'entities_info': entities_info,
                   'sentences_info': sentences_info,
                   'nodes_adj': nodes_adj,
                   }
        features.append(feature)
        
    json.dump(features, open(file_out, "w"))
    if "dwie" in file_in:
        logger.write("# of self-relation {}.\n".format(self_relation_num))
        logger.write("# of 1/0 entities {}.\n".format(single_or_no_entity_num))
    logger.write("# of documents {}.\n".format(i_line))
    logger.write("# of positive examples {}.\n".format(pos_samples))
    logger.write("# of negative examples {}.\n".format(neg_samples))

    return features


def create_graph(mentions_info, entities_info, sentences_info):
    # entity_node --> mention_node --> sentence_node
    mentions_num = len(mentions_info)
    entities_num = len(entities_info)
    sentences_num = len(sentences_info)

    node_num = sentences_num + mentions_num + entities_num
    # print(node_num, sentences_num, mentions_num, entities_num)
    nodes_adj = np.zeros((node_num, node_num), dtype=np.int32)

    # 1. ss
    for i in range(sentences_num):
        i_idx = i + entities_num + mentions_num
        for j in range(i+1, sentences_num):
            j_idx = j + entities_num + mentions_num
            nodes_adj[i_idx, j_idx] = 1
            nodes_adj[j_idx, i_idx] = 1

    # 2. sm & 3. em
    sent2men = defaultdict(list)
    entity2men = defaultdict(list)
    for i in range(mentions_num):
        sent_id = mentions_info[i][-1] + entities_num + mentions_num
        m_node_id = i+entities_num
        sent2men[sent_id].append(m_node_id)
        nodes_adj[sent_id, m_node_id] = 2
        nodes_adj[m_node_id, sent_id] = 2

        entity_id = mentions_info[i][-2]
        entity2men[entity_id].append(m_node_id)
        nodes_adj[entity_id, m_node_id] = 3
        nodes_adj[m_node_id, entity_id] = 3

    # 4. co
    for m_set in entity2men.values():
        for i in range(len(m_set)):
            for j in range(i+1, len(m_set)):
                nodes_adj[m_set[i], m_set[j]] = 4
                nodes_adj[m_set[j], m_set[i]] = 4

    # 5. mm
    for m_set in sent2men.values():
        for i in range(len(m_set)):
            for j in range(i+1, len(m_set)):
                nodes_adj[m_set[i], m_set[j]] = 5
                nodes_adj[m_set[j], m_set[i]] = 5

    # 6. es 
    for i in range(entities_num):
        entity_id = i 
        entity_m_list = entity2men[entity_id]
        for j in range(sentences_num):
            sent_id = j + entities_num + mentions_num
            sent_m_list = sent2men[sent_id]
            inter_set = list(set(entity_m_list) & set(sent_m_list))
            if len(inter_set) != 0:
                nodes_adj[entity_id, sent_id] = 6
                nodes_adj[sent_id, entity_id] = 6

    return nodes_adj.tolist()

