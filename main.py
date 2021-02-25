import argparse
import importlib
from src.evaluate import TREC_Eval_Command_Experiment
from src.BERT_transforms import MonoBERT_ReRanker_Transform, DuoBERT_ReRanker_Transform
from src.BM25_transforms import BM25_Search_Transform
from src.text_transforms import CAsT_Index_store, Base_Info_Transform, Baselies_Info_Transform
from src.utils import get_data, Info_Plotter

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import os
import torch
from dotmap import DotMap

module = importlib.import_module('src.models')
wandb_id = wandb.util.generate_id()

#############################
# Example usage
# $ python3 main.py --rewriter OracleReWriter --hits 100
#############################

parser = argparse.ArgumentParser(description='Fine Tune a query re-writing model for CAsT')
parser.add_argument('-d','--dataset', action='append', help='datasource to use for fine-tuning')
parser.add_argument('--rewriter', type=str, default="BART_ReWriter")
parser.add_argument('--use_sep', type=bool, default=False)
parser.add_argument('--use_doc_context', type=bool, default=False)
parser.add_argument('--decoding_format', type=str, default="last_turn")
parser.add_argument('--prev_queries_input', type=str, default="raw")
parser.add_argument('--skip_train', default=False, action='store_true')
parser.add_argument('--skip_neural_rerank', default=False, action='store_true')
parser.add_argument('--from_checkpoint', type=str, default='')
parser.add_argument('--num_eval_samples', type=int, default=-1)
parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id())
parser.add_argument('--name', type=str, default=None)

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_sz', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--hits', type=int, default=1000)
parser.add_argument('--duo_rerank_num', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--inference_chunk_size', type=int, default=32)
parser.add_argument('--save_dir', type=str, default='checkpoints')

base_args = DotMap()


def main(args):
    seed_everything(args.seed)
    # useful functions
    get_doc_fn = CAsT_Index_store().get_doc

    fine_tuning_samples = get_data(args.dataset, 'train')
    
    fine_tuning_samples = Base_Info_Transform()(fine_tuning_samples)
    fine_tuning_samples = Baselies_Info_Transform()(fine_tuning_samples)

    model_name = args.name if args.name else f"{args.rewriter}_data_{'_'.join(args.dataset)}_skipNR_{args.skip_neural_rerank}_epochs_{args.epochs}_batchSz_{args.batch_sz}_lr_{args.lr}_wandbID_{args.wandb_id}"
    logger = WandbLogger(name=model_name,project='CAsT_query_rewriting', id=args.wandb_id)


    re_writer_class = getattr(module, args.rewriter)
    print(f"Loading: {args.rewriter}")
    re_writer = re_writer_class(args)



    if args.from_checkpoint != '':
        print(f"Loading checkpoint for {args.rewriter} from {args.from_checkpoint}")
        re_writer.load_state_dict(torch.load((args.from_checkpoint)))
        re_writer.to(f"cuda:{args.gpu_id}")


    # train here or skip if indicated
    if not args.skip_train:
        dataloader = DataLoader(fine_tuning_samples, batch_size=args.batch_sz, num_workers=0, shuffle=True, collate_fn = re_writer.collate)
        lr_logger_cb = LearningRateMonitor(logging_interval='step')
        pl_trainer = Trainer(gpus=args.gpu_id, gradient_clip_val=0.5, amp_level='O1', max_epochs=args.epochs, logger=logger, callbacks=[lr_logger_cb])
        re_writer.train()
        pl_trainer.fit(re_writer, dataloader)
        re_writer.to(f"cuda:{args.gpu_id}")

        checkpoint_path = f"checkpoints/{model_name}.state_dict"
        print(f"Saving final model to: {checkpoint_path}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(re_writer.state_dict(), checkpoint_path)
    # end training






    # Evaluate model
    bm25_transform = BM25_Search_Transform(hits=args.hits, 
                                           key_fields={'query_field':'re-write', 'target_field':'search_results'})
    results_field = 'search_results'

    test_samples = get_data(['cast_y2'], 'test')[:args.num_eval_samples]
    test_samples = Base_Info_Transform()(test_samples) # adding raw and manual queries
    test_samples = Baselies_Info_Transform()(test_samples) # adding baseline system rewrites

    test_samples = re_writer.inference(test_samples, chunk_size=args.inference_chunk_size) # each sample should have a "re-write" field

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
                                                   rerank_top=args.duo_rerank_num,
                                                   device=f"cuda:{args.gpu_id}",
                                                   batch_size=args.inference_chunk_size,
                                                   key_fields={'query_field':'re-write', 
                                                               'source_field':'mono_rerank_results',
                                                               'target_field':'duo_rerank_results'})
        test_samples = monoBERT_transform(test_samples)
        test_samples = duoBERT_transform(test_samples)
        results_field = 'duo_rerank_results'
    
    eval_experiment = TREC_Eval_Command_Experiment(save_run_path=f"runs/{model_name}.run", save_eval_path=f"evals/{model_name}.eval", key_fields={'source_field':results_field})
    test_samples = eval_experiment(test_samples)

    for sample in test_samples:
        raw_query = sample['all_raw_queries'][-1]
        print(f"Q_id -> {sample['q_id']}")
        if eval_experiment.relevant_metrics[0] in sample:
            print('; '.join([f"{metric}: {sample[metric]}" for metric in eval_experiment.relevant_metrics]))
        else:
            print('NO Q_REL AVAILABLE so no metrics, my man...')
        print(f"{raw_query} -> {sample['re-write']}")
        print(f"{' '*len(raw_query)}   [{sample['all_manual_queries'][-1]}]")
        print(f"TOP DOCS")
        print()
        for doc_id, score in sample[results_field][:3]:
            print(f"Score: {score}")
            print(get_doc_fn(doc_id))
            print("------------------------------------------------------------------")
            print()
        print("############################")
        
    info_plotter = Info_Plotter()
    
    turn_plot_dict = info_plotter.per_turn_plots(test_samples)
    logger.log_metrics(turn_plot_dict)
    
    rewrites_dict = info_plotter.tabulate_rewrites(test_samples)
    logger.log_metrics(rewrites_dict)

    print("############ OVERALL ############")
    eval_dict = eval_experiment.overall(test_samples)
    scores = [f"{eval_dict[m]:0.3f}" for m in eval_experiment.relevant_metrics]
    print('\t'.join(eval_experiment.relevant_metrics))
    print('\t\t'.join(scores))

    logger.log_metrics(eval_dict)
    print("#################################")

    print("Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)