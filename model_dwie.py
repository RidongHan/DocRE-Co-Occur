import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from graph import GraphReasonLayer


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.bce_loss_fnt = nn.BCEWithLogitsLoss(reduction='none')

        fea_dim = self.hidden_size

        if config.joint_label_embed:
            fea_dim += self.hidden_size
            if config.rel_correl:
                self.correl_rel_repres_trans = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
                self.correl_rel_loss_fnt = nn.BCEWithLogitsLoss(reduction='none')

            if config.triplet_correl:
                self.correl_triplet_repres_trans = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
                self.correl_triplet_loss_fnt = nn.BCEWithLogitsLoss(reduction='none')

        if self.config.use_graph:
            self.graph_reason = GraphReasonLayer(
                ['ss', 'sm', 'em', 'co', 'mm', 'es'],
                self.hidden_size,
                self.hidden_size,
                2,
                graph_type=self.config.graph_type,
                graph_drop=0.5,
            )
            fea_dim += 3 * self.hidden_size
        else:
            fea_dim += self.hidden_size
        self.head_extractor = nn.Linear(fea_dim, self.hidden_size)
        self.tail_extractor = nn.Linear(fea_dim, self.hidden_size)
        self.bilinear = nn.Linear(self.hidden_size * block_size, config.num_labels)      
            

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention, pooler_output = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention, pooler_output


    def get_hrt(self, sequence_output, attention, entity_pos, hts, token2rel_att=None, relation_matrix=None):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0  # [CLS] or <s> 
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        if token2rel_att is not None:
            lss = []
        ht_num_list = []
        nodes_m_e = []  # graph node repre
        for i in range(len(entity_pos)):  # bs
            cur_mentions = []  # graph node repre
            cur_entities = []  # graph node repre
            entity_embs, entity_atts = [], []
            if token2rel_att is not None:
                entity2rel_atts = []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    if token2rel_att is not None:
                        e2l_att = []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            if token2rel_att is not None:
                                e2l_att.append(token2rel_att[i, :, start + offset])
                            cur_mentions.append(sequence_output[i, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                        if token2rel_att is not None:
                            e2l_att = torch.stack(e2l_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        if token2rel_att is not None:
                            e2l_att = torch.zeros(h, self.config.num_labels).to(token2rel_att)
                    cur_entities.append(e_emb)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                        if token2rel_att is not None:
                            e2l_att = token2rel_att[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        if token2rel_att is not None:
                            e2l_att = torch.zeros(h, self.config.num_labels).to(token2rel_att)
                    cur_mentions.append(e_emb)
                    cur_entities.append(e_emb)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                if token2rel_att is not None:
                    entity2rel_atts.append(e2l_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            if token2rel_att is not None:
                entity2rel_atts = torch.stack(entity2rel_atts, dim=0)  # [n_e, h, 97]

            cur_mentions = torch.stack(cur_mentions, dim=0)
            cur_entities = torch.stack(cur_entities, dim=0)
            cur_m_e = torch.cat([cur_entities, cur_mentions], dim=0)
            nodes_m_e.append(cur_m_e)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

            if token2rel_att is not None:
                h_att = torch.index_select(entity2rel_atts, 0, ht_i[:, 0])
                t_att = torch.index_select(entity2rel_atts, 0, ht_i[:, 1])
                ht_att = (h_att * t_att).mean(1)
                ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
                ls = contract("ld,rl->rd", relation_matrix[i], ht_att)
                lss.append(ls)

            ht_num_list.append(hs.shape[0])
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        if token2rel_att is  not None:
            lss = torch.cat(lss, dim=0)
            return hss, rss, tss, ht_num_list, nodes_m_e, lss
        else:
            return hss, rss, tss, ht_num_list, nodes_m_e
            

    def get_label_matrix_and_attention(self, sequence_output, attention, label_start_ids):
        label_matrix_list = []
        label_attention_list = []
        for i, idx in enumerate(label_start_ids):
            cur_matrix = sequence_output[i, idx: idx+self.config.num_labels, :].unsqueeze(0)
            label_matrix_list.append(cur_matrix)
            cur_att = attention[i, :, :, idx: idx+self.config.num_labels]
            # cur_att = cur_att/sum(cur_att + 1e-8)
            label_attention_list.append(cur_att)
        return torch.cat(label_matrix_list, dim=0), torch.stack(label_attention_list, dim=0)


    def get_train_data_for_label_correl_learn_mean(self, labels_set, pooler_output, relation_matrix):

        repres_for_correl_learn = []
        labels_for_correl_learn = []
        labels_mask_for_correl_learn = []

        for bs_id, la_list in enumerate(labels_set):
            la_list.sort()
            num_la_list = len(la_list)
            # 关系数目 小于2，跳过
            if num_la_list <= 1:  # no relation or one relation
                repres_for_correl_learn.append([])
                continue

            # 关系数目 大于等于2
            cur_relation_matrix = relation_matrix[bs_id]
            # cur_relation_matrix = torch.relu(self.correl_rel_trans(cur_relation_matrix))
            cur_repre_for_correl_learn = []
            for i in range(num_la_list):
                rest_la_list = la_list[:i] if (i+1)==num_la_list else la_list[:i] + la_list[i+1:] 

                cur_att_mask = torch.tensor([0.0] * self.config.num_labels).float().to(relation_matrix.device)
                cur_att_mask[rest_la_list] = 1.0
                cur_repre_for_correl_learn.append(torch.sum(cur_relation_matrix * cur_att_mask.unsqueeze(dim=1), dim=0) / torch.sum(cur_att_mask).float())

                cur_correl_label = torch.tensor([0] * self.config.num_labels).float().to(relation_matrix.device)
                cur_correl_label[la_list] = 1
                labels_for_correl_learn.append(cur_correl_label)

                cur_correl_label_mask = torch.tensor([0.0] + [1.0] * (self.config.num_labels-1)).float().to(relation_matrix.device)  # exclude NA
                cur_correl_label_mask[rest_la_list] = 0.0
                labels_mask_for_correl_learn.append(cur_correl_label_mask)

            cur_repre_for_correl_learn = torch.stack(cur_repre_for_correl_learn, dim=0)
            repres_for_correl_learn.append(cur_repre_for_correl_learn)

        labels_for_correl_learn = torch.stack(labels_for_correl_learn, dim=0) if labels_for_correl_learn != [] else []
        labels_mask_for_correl_learn = torch.stack(labels_mask_for_correl_learn, dim=0) if labels_mask_for_correl_learn != [] else []

        return repres_for_correl_learn, labels_for_correl_learn, labels_mask_for_correl_learn


    def get_train_data_for_triplet_correl_learn(self, htrs, pooler_output, labels_mask, ht_num_list):

        repres_for_triplet_correl = []
        labels_for_triplet_correl = []
        labels_mask_for_triplet_correl = []
        htrs_list = []

        cur_index = 0
        for i, ht_n in enumerate(ht_num_list):  # i: batch_id
            cur_label_mask = labels_mask[i]
            cur_htrs = htrs[cur_index: cur_index+ht_n]
            cur_index += ht_n

            cur_num_triplet = int(sum(cur_label_mask))

            # 实体对 小于 2对，跳过
            if cur_num_triplet <= 1:
                repres_for_triplet_correl.append([])
                htrs_list.append([])
                labels_mask_for_triplet_correl.append([])
                labels_for_triplet_correl.append([])
                continue

            # 实体对 大于 2 对
            htrs_list.append(cur_htrs)
            cur_htrs_no_na = cur_htrs[:cur_num_triplet]
            cur_htrs_id_list = list(range(cur_num_triplet))

            cur_repre_for_triplet_correl = []
            cur_labels_for_triplet_correl = []
            cur_labels_mask_for_triplet_correl = []
            for j in range(cur_num_triplet):
                rest_htrs_id_list = cur_htrs_id_list[:j] if (j+1)==cur_num_triplet else cur_htrs_id_list[:j] + cur_htrs_id_list[j+1:] 

                cur_att_mask = torch.tensor([0.0] * cur_num_triplet).float().to(pooler_output.device)
                cur_att_mask[rest_htrs_id_list] = 1.0
                cur_repre_for_triplet_correl.append(torch.sum(cur_htrs_no_na * cur_att_mask.unsqueeze(1), dim=0) / torch.sum(cur_att_mask))

                cur_correl_label = torch.tensor([0] * ht_n).float().to(pooler_output.device)
                cur_correl_label[cur_htrs_id_list] = 1
                cur_labels_for_triplet_correl.append(cur_correl_label)

                cur_correl_label_mask = torch.tensor([1.0] * ht_n).float().to(pooler_output.device)
                cur_correl_label_mask[rest_htrs_id_list] = 0.0
                cur_labels_mask_for_triplet_correl.append(cur_correl_label_mask)


            cur_repre_for_triplet_correl = torch.stack(cur_repre_for_triplet_correl, dim=0)
            repres_for_triplet_correl.append(cur_repre_for_triplet_correl)

            cur_labels_for_triplet_correl = torch.stack(cur_labels_for_triplet_correl, dim=0)
            labels_for_triplet_correl.append(cur_labels_for_triplet_correl)

            cur_labels_mask_for_triplet_correl = torch.stack(cur_labels_mask_for_triplet_correl, dim=0)
            labels_mask_for_triplet_correl.append(cur_labels_mask_for_triplet_correl)

        return repres_for_triplet_correl, htrs_list, labels_for_triplet_correl, labels_mask_for_triplet_correl

    
    def get_nodes_and_adjmatrix(self, sequence_output, sentences_info, nodes_adj, nodes_m_e):
        max_token_length = self.config.max_seq_length - self.config.num_class - 1 if self.config.joint_label_embed else self.config.max_seq_length
        nodes_sme = []

        for b_id, sent_info in enumerate(sentences_info):
            cur_s = []
            for sent in sent_info:
                start = sent[0]
                end = sent[1] if sent[1] < max_token_length else max_token_length-1
                sent_hidden = sequence_output[b_id][start: end] 
                sent_repre = torch.logsumexp(sent_hidden, dim=0)
                cur_s.append(sent_repre)
            cur_s = torch.stack(cur_s, dim=0)
            cur_m_e = nodes_m_e[b_id]
            cur_s_m_e = torch.cat([cur_m_e, cur_s], dim=0)
            nodes_sme.append(cur_s_m_e)
            assert cur_s_m_e.shape[0] == len(nodes_adj[b_id]), "[Error]: the number of nodes dismatch"
        max_num_node = max([nodes.shape[0] for nodes in nodes_sme])
        nodes_sme = [F.pad(nodes, (0, 0, 0, max_num_node-nodes.shape[0])) for nodes in nodes_sme]
        nodes_adj = [
            F.pad(
                torch.tensor(nodes).long().to(sequence_output.device), 
                (0, max_num_node-len(nodes), 0, max_num_node-len(nodes))
            ) 
            for nodes in nodes_adj
        ]
        nodes_sme = torch.stack(nodes_sme, dim=0)
        nodes_adj = torch.stack(nodes_adj, dim=0)
        # nodes_adj = (nodes_adj > 0).float() + torch.eye(nodes_adj.shape[-1]).unsqueeze(0).to(nodes_adj.device)  # 不区分 edges

        return nodes_sme, nodes_adj


    def get_ht_emb_after_graph(self, nodes_emb, entity_pos, hts):
        # offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0  # [CLS] or <s> 
        hss, tss = [], []
        for i in range(len(entity_pos)):  # bs
            entity_embs = nodes_emb[i]
            ht_i = torch.LongTensor(hts[i]).to(nodes_emb.device)
            
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)

        return hss, tss
                

    def forward(self, input_ids=None, input_mask=None, labels=None, entity_pos=None, hts=None, 
                labels_mask=None, labels_set=None, label_start_ids=None, sentences_info=None, nodes_adj=None, instance_mask=None):

        sequence_output, attention, pooler_output = self.encode(input_ids.to(self.config.device), input_mask.to(self.config.device))
        if self.config.joint_label_embed:
            relation_matrix, token2rel_att = self.get_label_matrix_and_attention(sequence_output, attention, label_start_ids)
            hs, rs, ts, ht_num_list, nodes_e_m, lss = self.get_hrt(sequence_output, attention, entity_pos, hts, token2rel_att, relation_matrix)
            hs_fea = [hs, rs, lss]
            ts_fea = [ts, rs, lss]
        else:
            hs, rs, ts, ht_num_list, nodes_e_m = self.get_hrt(sequence_output, attention, entity_pos, hts)
            hs_fea = [hs, rs]
            ts_fea = [ts, rs]

        if self.config.use_graph:
            nodes_emb, nodes_adj = self.get_nodes_and_adjmatrix(sequence_output, sentences_info, nodes_adj, nodes_e_m)
            nodes_emb = self.graph_reason(nodes_emb, nodes_adj)
            new_hs, new_ts = self.get_ht_emb_after_graph(nodes_emb, entity_pos, hts)
            hs_fea = hs_fea[1:]
            hs_fea.insert(0, new_hs)
            ts_fea = ts_fea[1:]
            ts_fea.insert(0, new_ts)

        hs_fea = torch.cat(hs_fea, dim=1)
        ts_fea = torch.cat(ts_fea, dim=1) 
        hs = torch.tanh(self.head_extractor(hs_fea))
        ts = torch.tanh(self.tail_extractor(ts_fea))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)  # [sum_all_hts, num_labels]
        
        output = {"logits": logits}

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)  # [sum_hts, 97]
            loss = self.bce_loss_fnt(logits.float(), labels.float()).mean()
            output["re_loss"] = loss

            if self.config.joint_label_embed:
                if self.config.rel_correl:
                    repres_for_correl_learn, labels_for_correl_learn, labels_mask_for_correl_learn = \
                        self.get_train_data_for_label_correl_learn_mean(labels_set, pooler_output, relation_matrix)

                    correl_logits = []
                    for idx, repre_for_correl_learn in enumerate(repres_for_correl_learn):
                        if repre_for_correl_learn != []:
                            correl_tmp_repre = self.correl_rel_repres_trans(repre_for_correl_learn)
                            cur_logit = torch.matmul(correl_tmp_repre, relation_matrix[idx].t())
                            correl_logits.append(cur_logit)
                    correl_logits = torch.cat(correl_logits, dim=0) if correl_logits != [] else []  # 二分类： 0: NotCo-Occur， 1: IsCo-Occur

                    if correl_logits != []:    
                        correl_loss = self.correl_rel_loss_fnt(correl_logits.float(), labels_for_correl_learn.float())# .mean()
                        correl_loss = torch.sum(correl_loss * labels_mask_for_correl_learn) / torch.sum(labels_mask_for_correl_learn)
                        assert not torch.isnan(correl_loss), "[Error]: Correl_loss is nan!!!"
                    else:
                        correl_loss = torch.tensor(0.0001).float().to(logits.device)  # mini-batch 里 全是 0/1个关系
                    output["correl_rel_loss"] = correl_loss

                if self.config.triplet_correl:
                    repres_for_triplet_correl, htrs_list, labels_for_triplet_correl, labels_mask_for_triplet_correl = \
                            self.get_train_data_for_triplet_correl_learn(lss, pooler_output, labels_mask, ht_num_list)

                    correl_triplet_loss = []
                    for idx, repre_for_triplet_correl in enumerate(repres_for_triplet_correl):
                        if repre_for_triplet_correl != []:
                            correl_tmp_repre = self.correl_triplet_repres_trans(repre_for_triplet_correl)
                            cur_logit = torch.matmul(correl_tmp_repre, htrs_list[idx].t())  # 二分类： 0: NotCo-Occur， 1: IsCo-Occur

                            cur_tmp_loss = self.correl_triplet_loss_fnt(cur_logit.float(), labels_for_triplet_correl[idx].float())
                            cur_tmp_loss = torch.sum(cur_tmp_loss * labels_mask_for_triplet_correl[idx]) / torch.sum(labels_mask_for_triplet_correl[idx])
                            assert not torch.isnan(cur_tmp_loss), "[Error]: correl_triplet_loss is nan!!!"
                            correl_triplet_loss.append(cur_tmp_loss)
                    
                    correl_triplet_loss = torch.stack(correl_triplet_loss, dim=0).mean()
                    output["correl_triplet_loss"] = correl_triplet_loss

        return output

