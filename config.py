import argparse, os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument('--prepro_dir', type=str, default="./processed_data/docred")
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument('--zigzag', action='store_true', default=False)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="./checkpoint/docred", type=str)
    parser.add_argument('--model_prefix', type=str, default="")
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    parser.add_argument('--use_graph', action='store_true', default=False)
    parser.add_argument("--graph_type", type=str, default="", help="gcn or gat")

    parser.add_argument('--triplet_correl', action='store_true', default=False)
    parser.add_argument('--rel_correl', action='store_true', default=False)
    parser.add_argument('--joint_label_embed', action='store_true', default=False)
    parser.add_argument("--alpha", default=0.7, type=float)

    args = parser.parse_args()

    if args.joint_label_embed:
        args.prepro_dir = os.path.join(args.prepro_dir, "JE/")
    
    if args.joint_label_embed:
        if args.triplet_correl and args.rel_correl:
            args.save_path = os.path.join(args.save_path, "je_tri_rel")
        elif args.rel_correl:
            args.save_path = os.path.join(args.save_path, "je_rel")
        elif args.triplet_correl:
            args.save_path = os.path.join(args.save_path, "je_tri")

    if not os.path.exists(args.prepro_dir):
        os.makedirs(args.prepro_dir)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args