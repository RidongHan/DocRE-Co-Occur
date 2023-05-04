import os, time
from tabnanny import check
import random
import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel as DocREDModel
from model_dwie import DocREModel as DWIEModel
from utils import set_seed, collate_fn, Logger
from prepro import read_docred
from evaluation import to_official_overlap, official_evaluate, macro_evaluate, overlap_evaluate, to_official, strict_overlap_evaluate
from config import get_args
from torch.utils.tensorboard import SummaryWriter


def get_output_labels(logits, theta=0.):
    output = torch.zeros_like(logits).to(logits.device) 
    mask = (torch.sigmoid(logits)>theta)
    output[mask] = 1.0
    output[:, 0] = (output.sum(dim=1) == 0).to(logits.device)
    return output


def evaluate(args, model, features, id2rel, logger, tag="dev", g_threshold=None, macro=False, use_g_thres=False):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        with torch.no_grad():
            batch.pop('labels')
            outputs = model(**batch)
            pred = outputs["logits"]
            preds.append(pred)
    
    preds = torch.cat(preds, dim=0)

    if tag == "dev":
        best_f1 = 0.0
        best_f1_ign = 0.0
        best_threshold = 0.0

        global_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        logger.write("-"*80)
        logger.write("\nEvaluation with different global thereshold ...\n")
        for threshold in global_threshold:
            cur_output = get_output_labels(preds, theta=threshold).cpu().numpy()
            cur_output[np.isnan(cur_output)] = 0
            cur_output = cur_output.astype(np.float32)

            ans = to_official(cur_output, features, id2rel)
            cur_f1 = 0.0
            cur_f1_ign = 0.0
            if len(ans) > 0:
                cur_f1, _, cur_f1_ign, _ = official_evaluate(ans, args.data_dir, eval=tag)
            cur_output = {
                tag + "_F1": cur_f1 * 100,
                tag + "_F1_ign": cur_f1_ign * 100,
            }
            logger.write(f"[Threshold {threshold}]: ")
            logger.write(json.dumps(cur_output) + "\n")
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_f1_ign = cur_f1_ign
                best_threshold = threshold
            # if len(ans) > 0:
            #     overlap_output = overlap_evaluate(ans, args.data_dir, eval=tag)
            #     logger.write(json.dumps(overlap_output, indent=4) + "\n")
        logger.write("-"*80 + "\n")

        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            tag + "_Threshold": best_threshold,
        }
        overlap_output = evaluate_overlap(args, preds, best_threshold, features, id2rel, tag="dev", logger=logger)

        return best_f1, best_threshold, output


def evaluate_overlap(args, preds, best_threshold, features, id2rel, tag="dev", logger=None):
    cur_output = get_output_labels(preds, theta=best_threshold).cpu().numpy()
    cur_output[np.isnan(cur_output)] = 0
    cur_output = cur_output.astype(np.float32)

    ans = to_official(cur_output, features, id2rel)
    if len(ans) > 0:
        overlap_output = overlap_evaluate(ans, args.data_dir, eval=tag)
        logger.write(json.dumps(overlap_output, indent=4) + "\n")
    return cur_output


def main():
    args = get_args()
    
    log_file = "Eval-" + args.data_dir.split("/")[-1] + "-" + args.load_path.split("/")[-1].split(".")[0] + ".log"
    logger = Logger(file_name=log_file, log=False)
    logger.write(json.dumps(args.__dict__, indent=4))
    logger.write("\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if args.joint_label_embed:  # joint learn label embeddings
        rel_list = []
        for i in range(args.num_class):
            rel_list.append("[rel-" + str(i) + "]")
        tokenizer.add_tokens(rel_list, special_tokens=True)

    rel2id = json.load(open(os.path.join(args.data_dir, 'rel2id.json'), 'r'))
    id2rel = {value: key for key, value in rel2id.items()}
    read = read_docred

    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    dev_file_out = os.path.join(args.prepro_dir, args.transformer_type + "_" + args.dev_file)
    test_file_out = os.path.join(args.prepro_dir, args.transformer_type + "_" + args.test_file)

    dev_features = read(args, dev_file, dev_file_out, tokenizer, rel2id, max_seq_length=args.max_seq_length, logger=logger)
    test_features = read(args, test_file, test_file_out, tokenizer, rel2id, max_seq_length=args.max_seq_length, logger=logger)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    if args.joint_label_embed:
        model.resize_token_embeddings(len(tokenizer))

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.max_seq_length = args.max_seq_length
    config.num_class = args.num_class
    config.use_graph = args.use_graph
    config.graph_type = args.graph_type
    config.device = args.device
    config.triplet_correl = args.triplet_correl
    config.rel_correl = args.rel_correl
    config.joint_label_embed = args.joint_label_embed
    
    if "docred" in args.data_dir:
        model = DocREDModel(config, model, num_labels=args.num_labels)
    elif "dwie" in args.data_dir:
        model = DWIEModel(config, model, num_labels=args.num_labels)
    model.to(0)
    
    print(model)
    logger.write('total parameters:' + str(sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad])) + "\n")

    if args.load_path == "":  # Error
        logger.write('Load_path cannot be empty!!!')
    else:  # Testing
        args.load_path = os.path.join(args.save_path, args.load_path)
        check_point = args.load_path.split("/")[-1]
        logger.write(f"evaluation begins for checkpoint: {check_point}\n")
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_path), strict=False)
        
        dev_score, best_thres, dev_output = evaluate(args, model, dev_features, id2rel, logger, tag="dev", macro=True)
        logger.write(json.dumps(dev_output) + " | time: " + str(time.time()-start_time) + "s\n")


if __name__ == "__main__":
    main()
