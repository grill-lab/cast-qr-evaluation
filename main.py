import argparse
import importlib
from src.evaluate import TREC_Eval_Command_Experiment
from src.BERT_transforms import MonoBERT_ReRanker_Transform, DuoBERT_ReRanker_Transform
from src.BM25_transforms import BM25_Search_Transform
from src.text_transforms import CAsT_Index_store

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
parser.add_argument('--skip_train', type=bool, default=False)
parser.add_argument('--save_run_path', type=str, default='runs/latest.run')
parser.add_argument('--save_eval_path', type=str, default='evals/latest.eval')

parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--batch_sz', type=int, default=32)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--hits', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default='checkpoints')

args = parser.parse_args()

# useful functions
get_doc_fn = CAsT_Index_store().get_doc

# fine_tuning_samples = get_data(['CAsT', 'CANARD'], 'train')

re_writer_class = getattr(module, args.rewriter)
re_writer = re_writer_class()






# train here or skip if indicated

# end training






# Evaluate model
eval_experiment = TREC_Eval_Command_Experiment(save_run_path=args.save_run_path, save_eval_path=args.save_eval_path, key_fields={'source_field':'duo_rerank_results'})

bm25_transform = BM25_Search_Transform(hits=args.hits, 
                                       key_fields={'query_field':'re-write', 'target_field':'search_results'})

monoBERT_transform = MonoBERT_ReRanker_Transform('big_files/monoBERT', 
                                                 get_doc_fn, 
                                                 device=args.device,
                                                 key_fields={'query_field':'re-write', 'source_field':'search_results', 'target_field':'mono_rerank_results'})

duoBERT_transform = DuoBERT_ReRanker_Transform('big_files/duoBERT', 
                                               get_doc_fn, 
                                               device=args.device,
                                               key_fields={'query_field':'re-write', 'source_field':'mono_rerank_results', 'target_field':'duo_rerank_results'})

test_samples = [{
    "q_id":'81_2', 
    "all_raw":[
        "How do you know when your garage door opener is going bad?",
        "Now it stopped working. Why?"],
    "all_man":[
        "How do you know when your garage door opener is going bad?",
        "Now my garage door opener stopped working. Why?"],
}, {
    "q_id":'81_3', 
    "all_raw":[
        "How do you know when your garage door opener is going bad?",
        "Now it stopped working. Why?",
        "How much does it cost for someone to fix it?"],
    "all_man":[
        "How do you know when your garage door opener is going bad?",
        "Now my garage door opener stopped working. Why?",
        "How much does it cost for someone to repair a garage door opener?"],
}]

test_samples = re_writer.inference(test_samples) # each sample should have a "re-write" field

test_samples = bm25_transform(test_samples)
test_samples = monoBERT_transform(test_samples)
test_samples = duoBERT_transform(test_samples)

test_samples = eval_experiment(test_samples)

print("### OVERALL ###")
for metric in eval_experiment.relevant_metrics:
    all_scores = [s[metric] for s in test_samples]
    avg = sum(all_scores)/len(all_scores)
    print(f"{metric}: {avg}")
print("########################")

for sample in test_samples[:3]:
    raw_query = sample['all_raw'][-1]
    print(f"Q_id -> {sample['q_id']}")
    print('; '.join([f"{metric}: {sample[metric]}" for metric in eval_experiment.relevant_metrics]))
    print(f"{raw_query} -> {sample['re-write']}")
    print(f"TOP DOCS")
    print()
    for doc_id, score in sample['duo_rerank_results'][:3]:
        print(f"Score: {score}")
        print(get_doc_fn(doc_id))
        print("------------------------------------------------------------------")
        print()
    print("############################")

print("Done!")