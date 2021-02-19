import argparse
import importlib
from src.evaluate import TREC_Eval_Command_Experiment
from src.BERT_transforms import MonoBERT_ReRanker_Transform, DuoBERT_ReRanker_Transform
from src.BM25_transforms import BM25_Search_Transform
from src.text_transforms import CAsT_Index_store
from src.utils import get_data

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, Callback, seed_everything

module = importlib.import_module('src.models')

#############################
# Example usage
# $ python3 main.py --rewriter OracleReWriter --hits 100
#############################

parser = argparse.ArgumentParser(description='Fine Tune a query re-writing model for CAsT')
parser.add_argument('-d','--dataset', action='append', help='datasource to use for fine-tuning')
parser.add_argument('--rewriter', type=str, default="BART")
parser.add_argument('--use_sep', type=bool, default=False)
parser.add_argument('--use_doc_context', type=bool, default=False)
parser.add_argument('--decoding_format', type=str, default="last_turn")
parser.add_argument('--prev_queries_input', type=str, default="raw")
parser.add_argument('--skip_train', default=False, action='store_true')
parser.add_argument('--skip_neural_rerank', default=False, action='store_true')
parser.add_argument('--save_run_path', type=str, default='runs/latest.run')
parser.add_argument('--save_eval_path', type=str, default='evals/latest.eval')
parser.add_argument('--num_eval_samples', type=int, default=-1)

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--batch_sz', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--hits', type=int, default=1000)
parser.add_argument('--inference_chunk_size', type=int, default=64)
parser.add_argument('--save_dir', type=str, default='checkpoints')

args = parser.parse_args()

# useful functions
get_doc_fn = CAsT_Index_store().get_doc

fine_tuning_samples = get_data(args.dataset, 'train')

re_writer_class = getattr(module, args.rewriter)
re_writer = re_writer_class()






# train here or skip if indicated
if not args.skip_train:
    dataloader = DataLoader(fine_tuning_samples, batch_size=args.batch_sz, num_workers=0, collate_fn = re_writer.collate)
    pl_trainer = Trainer(gpus=args.gpu_id, gradient_clip_val=0.5, amp_level='O1', max_epochs=args.epochs)
    re_writer.train()
    pl_trainer.fit(re_writer, dataloader)
    re_writer.to(f"cuda:{args.gpu_id}")
# end training






# Evaluate model
eval_experiment = TREC_Eval_Command_Experiment(save_run_path=args.save_run_path, save_eval_path=args.save_eval_path, key_fields={'source_field':'search_results'})

bm25_transform = BM25_Search_Transform(hits=args.hits, 
                                       key_fields={'query_field':'re-write', 'target_field':'search_results'})
results_field = 'search_results'

test_samples = get_data(['cast_y2'], 'test')[:args.num_eval_samples]

test_samples = re_writer.inference(test_samples) # each sample should have a "re-write" field

test_samples = bm25_transform(test_samples)

if not args.skip_neural_rerank:
    monoBERT_transform = MonoBERT_ReRanker_Transform('big_files/monoBERT',
                                                 get_doc_fn, 
                                                 device=f"cuda:{args.gpu_id}",
                                                 batch_size=args.inference_chunk_size,
                                                 key_fields={'query_field':'re-write',
                                                             'source_field':'search_results',
                                                             'target_field':'mono_rerank_results'})

    duoBERT_transform = DuoBERT_ReRanker_Transform('big_files/duoBERT', 
                                               get_doc_fn, 
                                               device=f"cuda:{args.gpu_id}",
                                               batch_size=args.inference_chunk_size,
                                               key_fields={'query_field':'re-write', 
                                                           'source_field':'mono_rerank_results',
                                                           'target_field':'duo_rerank_results'})
    test_samples = monoBERT_transform(test_samples)
    test_samples = duoBERT_transform(test_samples)
    results_field = 'duo_rerank_results'

test_samples = eval_experiment(test_samples)

for sample in test_samples:
    raw_query = sample['all_raw_queries'][-1]
    print(f"Q_id -> {sample['q_id']}")
    print('; '.join([f"{metric}: {sample[metric]}" for metric in eval_experiment.relevant_metrics]))
    print(f"{raw_query} -> {sample['re-write']}")
    print(f"TOP DOCS")
    print()
    for doc_id, score in sample[results_field][:3]:
        print(f"Score: {score}")
        print(get_doc_fn(doc_id))
        print("------------------------------------------------------------------")
        print()
    print("############################")
    
print("############ OVERALL ############")
print(eval_experiment.overall(test_samples))
print("#################################")

print("Done!")