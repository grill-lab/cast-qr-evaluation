from tqdm import tqdm
import os
import copy
import json

import wandb

class RUN_File_Transform_Exporter():
    def __init__(self, run_file_path, model_name='model_by_carlos', key_fields={'source_field':'search_results'}, **kwargs):
        '''
        A Transform Exporter that creates a RUN file from samples returnedd by a search engine.
        '''
        self.run_file_path = run_file_path
        self.model_name = model_name
        self.key_fields = key_fields
        self.model_outputs_path = f'{self.run_file_path}.json'
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.63)...]},...]
        '''
        
        with open(self.model_outputs_path, 'w') as out_file:
            out_lst = []
            for s in samples:
                new_obj = copy.deepcopy(s)
                if 'search_results' in new_obj:
                    del new_obj['search_results']
                if 'mono_rerank_results' in new_obj:
                    del new_obj['mono_rerank_results']
                if 'duo_rerank_results' in new_obj:
                    del new_obj['duo_rerank_results']
                out_lst.append(new_obj)
            json.dump(out_lst, out_file)
        
        total_samples = 0
        os.makedirs(os.path.dirname(self.run_file_path), exist_ok=True)
        with open(self.run_file_path, 'w') as run_file:
            for sample_obj in tqdm(samples, desc='Writing to RUN file', leave=False):
                q_id = sample_obj['q_id']
                search_results = sample_obj[self.key_fields['source_field']]
                ordered_results = sorted(search_results, key=lambda res: res[1], reverse=True)
                for idx, result in enumerate(ordered_results):
                    d_id, score = result
                    total_samples+=1
                    run_file.write(f"{q_id} Q0 {d_id} {idx+1} {score} {self.model_name}\n")
        print(f"Successfully written {total_samples} samples from {len(samples)} queries run to: {self.run_file_path}")

class TREC_Eval_Command_Experiment():
    def __init__(self, trec_eval_command='trec_eval -q -c -M1000  -m ndcg_cut.3,5,10,15,20,100,1000 -m all_trec qRELS RUN_FILE',
                relevant_metrics=['ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_1000', 'map_cut_1000', 'recall_500', 'recall_1000'],
                q_rel_file='small_files/2020qrels.txt', save_run_path='/tmp/temp_run_by_carlos.run', eval_path='evals/latest.eval', **kwargs):
        '''
        This is an experiment transform that uses the official trec_eval command to compute scores for each query 
        and return valid results according to the command specified.
        '''
        self.trec_eval_command = trec_eval_command
        self.relevant_metrics = relevant_metrics
        self.q_rel_file = q_rel_file
        self.eval_path = eval_path
        
        self.temp_run_file = save_run_path
        self.run_file_exporter = RUN_File_Transform_Exporter(self.temp_run_file, model_name='temp_model_by_carlos', **kwargs)
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.63)...]},...]
        returns: [dict]: [{'q_id':"xxx", 'search_results':[("MARCO_xxx", 0.63)...], 'ndcg_cut_3':0.33, 'ndcg_cut_5'...},...]
        '''
        self.run_file_exporter(samples)
        resolved_command = self.trec_eval_command.replace('qRELS', self.q_rel_file).replace('RUN_FILE', self.temp_run_file)
        print(f'Running the following command: {resolved_command} > {self.eval_path}')
        os.makedirs(os.path.dirname(self.eval_path), exist_ok=True)
        os.system(f'{resolved_command} > {self.eval_path}')
        
        with open(self.eval_path, 'r') as eval_f:
            eval_results = {}
            for row in eval_f:
                if not any([metric in row for metric in self.relevant_metrics]):
                    continue
                metric, q_id, score = row.split()
                if q_id not in eval_results:
                    eval_results[q_id] = {}
                eval_results[q_id][metric] = float(score)
        self.eval_results = eval_results
                
        for sample in samples:
            if sample['q_id'] not in eval_results:
                print(f"q_rel missing for q_id {sample['q_id']}. No scores added to sample")
                continue
            sample.update(eval_results[sample['q_id']])
        return samples
    
    def overall(self, samples):
        self(samples)
        return {m:s for m, s in self.eval_results['all'].items() if m in self.relevant_metrics}