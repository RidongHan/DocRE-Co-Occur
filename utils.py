import torch
import random
import numpy as np
import os, sys


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class Logger(object):
    def __init__(self, file_name = 'correl_re.log', log=True, stream = sys.stdout) -> None:
        self.terminal = stream
        self.log = log
        if log:
            log_dir = "./outlog"
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.log = open(f'outlog/{file_name}', "a")
            self.flush()

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)

    def flush(self):
        self.log.seek(0)	# 定位
        self.log.truncate()


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    labels_mask = [f["labels_mask"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    labels_set = [f["label_set"] for f in batch]
    label_start_ids = [f["label_start_ids"] for f in batch]
    sentences_info = [f["sentences_info"] for f in batch]
    nodes_adj = [f["nodes_adj"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)  # [bs, max_len]
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    output = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "labels": labels,
        "entity_pos": entity_pos,
        "hts": hts,
        "labels_mask": labels_mask,
        "labels_set": labels_set,
        "label_start_ids": label_start_ids,
        "sentences_info": sentences_info,
        "nodes_adj": nodes_adj
    }
    return output
