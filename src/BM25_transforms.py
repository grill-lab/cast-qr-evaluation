from pyserini.search import SimpleSearcher
from tqdm import tqdm

class BM25_Ranker():
    def __init__(self, index_dir="big_files/CAsT_collection_with_meta.index", k1=0.82, b=0.68, **kwargs):
        self.searcher = SimpleSearcher(index_dir)
        self.searcher.set_bm25(k1, b)
        
    def predict(self, query, hits=10, **kwargs):
        search_results = self.searcher.search(query, k=hits)
        len_res = min(hits, len(search_results))
        results = [(search_results[i].docid, search_results[i].score) for i in range(len_res)]
        return results
    
class BM25_Search_Transform():
    def __init__(self, hits=100, key_fields={'query_field':'query', 'target_field':'search_results'}, **kwargs):
        '''
        first_pass_model_fn: ("query text") -> [(d_id, score), ...]
        '''
        self.first_pass_model_fn = BM25_Ranker(**kwargs).predict
        self.hits = hits
        self.key_fields = key_fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query text", ...}]
        returns: [dict]: [{'query':"query text", 'search_results':[("MARCO_xxx", 0.4), ("CAR_xxx",0.3)..], ...}]
        '''
        for sample_obj in tqdm(samples, desc="Searching queries"):
            query = sample_obj[self.key_fields['query_field']]
            results = self.first_pass_model_fn(query, hits=self.hits)
            sample_obj[self.key_fields['target_field']] = results
        return samples