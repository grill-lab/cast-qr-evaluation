from src.text_transforms import Document_Resolver_Transform, DuoBERT_Numericalise_Transform, MonoBERT_Numericalise_Transform
from src.utils import chunks

import torch
from tqdm import tqdm
from transformers import BertModel, BertForSequenceClassification, BertConfig
from itertools import permutations 
from scipy.interpolate import interp1d

class BertForPassageRanking(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.autograd.Variable(torch.ones(2, config.hidden_size),
                                              requires_grad=True)
        self.bias = torch.autograd.Variable(torch.ones(2), requires_grad=True)


class monoBERT_Scorer_Transform():
    def __init__(self, checkpoint_dir="saved_models/monoBERT/", device=None, PAD_id=0, batch_size=32, **kwargs):
        '''
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        if device:
            self.device = device
        else:
            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading chekcpoint from {checkpoint_dir}")
        self.BERT_Reranker = BertForPassageRanking.from_pretrained(checkpoint_dir, from_tf=True)
        self.BERT_Reranker.classifier.weight.data = self.BERT_Reranker.weight.data
        self.BERT_Reranker.classifier.bias.data = self.BERT_Reranker.bias.data
        self.BERT_Reranker.eval()
        self.BERT_Reranker.to(self.device)
        self.batch_size = batch_size
        self.PAD = PAD_id
        
        print(f"MonoBERT ReRanker initialised on device {self.device}. Batch size {batch_size}")
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1]}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], "score":0.56}]
        '''
        all_scores = torch.zeros((0,1), device=self.device)
        for sample_obj_batch in chunks(samples, self.batch_size):
            with torch.no_grad():
                input_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["input_ids"], dtype=torch.long, device=self.device) for sample_obj in sample_obj_batch], 
                                                     padding_value=self.PAD).T
                type_ids = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["type_ids"], dtype=torch.long) for sample_obj in sample_obj_batch], 
                                                     padding_value=self.PAD).T.to(self.device)
                attention_mask = (input_tensor != self.PAD).type(torch.float).to(self.device)
                scores = self.BERT_Reranker(input_tensor, attention_mask=attention_mask, token_type_ids=type_ids)[0][:,1].tolist()
            for sample_obj, score in zip(sample_obj_batch, scores):
                sample_obj["score"] = score
        return samples
    
class DuoBERT_Scorer_Transform():
    def __init__(self,checkpoint_dir="./saved_models/duoBERT/", device=None, PAD_id=0, batch_size=32, **kwargs):
        '''
        DuoBERT takes in a query and two documents and gives a scoore to the one that is most rellevant between each.
        
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        if device:
            self.device = device
        else:
            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading chekcpoint from {checkpoint_dir}")
        self.duoBERT_Reranker = BertForPassageRanking.from_pretrained(checkpoint_dir, from_tf=True)
        self.duoBERT_Reranker.classifier.weight.data = self.duoBERT_Reranker.weight.data
        self.duoBERT_Reranker.classifier.bias.data = self.duoBERT_Reranker.bias.data
        type_embed_weight = self.duoBERT_Reranker.bert.embeddings.token_type_embeddings.weight.data
        self.duoBERT_Reranker.bert.embeddings.token_type_embeddings.weight.data = torch.cat((type_embed_weight, torch.zeros(1,1024)))
        self.duoBERT_Reranker.to(self.device)
        self.duoBERT_Reranker.eval()
        self.batch_size = batch_size
        self.PAD = PAD_id
        print(f"DuoBERT ReRanker initialised on {self.device}. Batch size {batch_size}")
    
    def __call__(self, samples):
        '''
        The score given corresponds to the likelihood A is more relevant than B. So I higher score is favorrable for A.
        
        samples: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], 'score':0.95, ...}]
        '''
        for sample_obj_batch in chunks(samples, self.batch_size):
            with torch.no_grad():
                input_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["input_ids"], dtype=torch.long) for sample_obj in sample_obj_batch], 
                                                     padding_value=self.PAD).T.to(self.device)
                type_ids = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["type_ids"], dtype=torch.long) for sample_obj in sample_obj_batch], 
                                                     padding_value=self.PAD).T.to(self.device)
                attention_mask = (input_tensor != self.PAD).type(torch.float).to(self.device)
                scores = outputs = self.duoBERT_Reranker(input_tensor, attention_mask=attention_mask, token_type_ids=type_ids)[0][:,1].tolist()
            for sample_obj, score in zip(sample_obj_batch, scores):
                sample_obj["score"] = score
        return samples
    
class MonoBERT_ReRanker_Transform():
    def __init__(self, checkpoint_dir, get_doc_fn, key_fields={'query_field':'query', 'source_field':'search_results', 'target_field':'reranked_results'}, **kwargs):
        '''
        A Transform that reorders a list based on BERT query doc score
        
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        self.monoBERT_score_transform = monoBERT_Scorer_Transform(checkpoint_dir, **kwargs)
        self.doc_resolver_transform = Document_Resolver_Transform(get_doc_fn)
        self.monoBERT_numericalise_transform = MonoBERT_Numericalise_Transform(**kwargs)
        self.key_fields = key_fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query text",'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..]...}]
        returns: [dict]: [{'query':"query text",'reranked_results':[("CAR_xxx", 0.54), ("CAR_xxx",0.27)..]...}]
        '''
        for sample_obj in tqdm(samples, desc="Reranking queries"):
            query = sample_obj[self.key_fields['query_field']]
            reranking_samples = [{'query':query, 'd_id':d_id} for d_id, score in sample_obj[self.key_fields['source_field']]]
            reranking_samples = self.doc_resolver_transform(reranking_samples)
            reranking_samples = self.monoBERT_numericalise_transform(reranking_samples)
            reranking_samples = self.monoBERT_score_transform(reranking_samples)
            ordered_samples = sorted(reranking_samples, key=lambda sample: sample['score'], reverse=True)
            sample_obj[self.key_fields['target_field']] = [(sample['d_id'], sample['score']) for sample in ordered_samples]
        return samples
    
class DuoBERT_ReRanker_Transform():
    def __init__(self, checkpoint_dir, get_doc_fn, rerank_top=10, key_fields={'query_field':'query', 'source_field':'search_results',  'target_field':'reranked_results'}, **kwargs):
        '''
        A Transform that reorders a list pairwise.
        
        checkpoint_path: str: path to only the state dict of the model, loaded with load_state_dict
        '''
        self.rerank_top = rerank_top
        self.duoBERT_score_transform = DuoBERT_Scorer_Transform(checkpoint_dir, **kwargs)
        self.doc_resolver_transform = Document_Resolver_Transform(get_doc_fn, fields=[('d_idA','docA'),('d_idB','docB')])
        self.duoBERT_numericalise_transform = DuoBERT_Numericalise_Transform(**kwargs)
        self.key_fields = key_fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query text",'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..]...}]
        returns: [dict]: [{'query':"query text",'duo_reranked_results':[("CAR_xxx", 0.54), ("CAR_xxx",0.27)..]...}]
        '''
        for sample_obj in tqdm(samples, desc="Reranking queries"):
            query = sample_obj[self.key_fields['query_field']]
            
            d_ids = [d_id for d_id, score in sample_obj[self.key_fields['source_field']]]
            d_id_permutations = list(permutations(d_ids[:self.rerank_top], 2))

            doc_permutations = [{'query':query, 'd_idA':d_idA, 'd_idB':d_idB} for d_idA, d_idB in d_id_permutations]
            doc_permutations = self.doc_resolver_transform(doc_permutations)
            doc_permutations = self.duoBERT_numericalise_transform(doc_permutations)
            scored_permutations = self.duoBERT_score_transform(doc_permutations)
            
            d_id_scores = {}
            for scored_perm in scored_permutations:
                if scored_perm['d_idA'] not in d_id_scores:
                    d_id_scores[scored_perm['d_idA']] = 0
                if scored_perm['d_idB'] not in d_id_scores:
                    d_id_scores[scored_perm['d_idB']] = 0
                    
                d_id_scores[scored_perm['d_idA']] += scored_perm['score']
                d_id_scores[scored_perm['d_idB']] -= scored_perm['score']
            
            new_scored_samples = [(d_id, score) for d_id, score in d_id_scores.items()]
            
            current_scores = [score for d_id, score in sample_obj[self.key_fields['source_field']][:self.rerank_top]]
            new_scores = [score for d_id, score in new_scored_samples]
            score_map = interp1d([min(new_scores),max(new_scores)],[min(current_scores),max(current_scores)])
            
            new_scored_samples = [(d_id, float(score_map(score))) for d_id, score in new_scored_samples]
            
            sample_obj[self.key_fields['target_field']] = sorted(new_scored_samples, key=lambda score_tpl: score_tpl[1], reverse=True)
            sample_obj[self.key_fields['target_field']] += [s for s in sample_obj[self.key_fields['source_field']] if s[0] not in d_id_scores]
            
        return samples