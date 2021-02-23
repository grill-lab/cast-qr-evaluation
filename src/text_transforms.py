from pyserini.search import SimpleSearcher
from transformers import BertTokenizer


class CAsT_Index_store():
    def __init__(self, CAsT_index="big_files/CAsT_collection_with_meta.index"):
        self.searcher = SimpleSearcher(CAsT_index)
    
    def get_doc(self, doc_id):
        raw_text = self.searcher.doc(doc_id).raw()
        paragraph = raw_text[raw_text.find('<BODY>\n')+7:raw_text.find('\n</BODY>')]
        return paragraph
    
class Base_Info_Transform():
    def __init(self):
        pass
    def __call__(self, samples):
        for sample_obj in samples:
            sample_obj["raw query"] = sample_obj['all_raw_queries'][-1]
            sample_obj["manual query"] = sample_obj['all_manual_queries'][-1]
        return samples
            
class Document_Resolver_Transform():
    def __init__(self, get_doc_fn, fields=[('d_id','doc')], **kwargs):
        '''
        get_doc_fn: fn(d_id) -> "document string"
        '''
        self.get_doc_fn = get_doc_fn
        self.fields = fields
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'d_id':"CAR_xxx", ...}]
        returns: [dict]: [{'doc':"document text", 'd_id':"CAR_xxx", ...}]
        '''
        for sample_obj in samples:
            for input_field, target_field in self.fields:
                sample_obj[target_field] = self.get_doc_fn(sample_obj[input_field])
        return samples
    
class DuoBERT_Numericalise_Transform():
    def __init__(self, vocab_txt_file="big_files/duoBERT/vocab.txt", **kwargs):
        self.numericalizer = BertTokenizer(vocab_txt_file)
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'query':"text and more", 'docA':"docA text", 'docB':"docB text" ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            query_text = sample_obj['query']
            query_ids = [self.numericalizer.cls_token_id]+self.numericalizer.encode(query_text, add_special_tokens=False)[:62]+[self.numericalizer.sep_token_id]
            query_token_type_ids = [0]*len(query_ids)
            
            docA_text = sample_obj['docA']
            docA_ids = self.numericalizer.encode(docA_text, add_special_tokens=False)[:223] + [self.numericalizer.sep_token_id]
            docA_token_type_ids = [1]*len(docA_ids)
            
            docB_text = sample_obj['docB']
            docB_ids = self.numericalizer.encode(docB_text, add_special_tokens=False)[:223] + [self.numericalizer.sep_token_id]
            docB_token_type_ids = [2]*len(docB_ids)
            
            sample_obj["input_ids"] = query_ids+docA_ids+docB_ids
            sample_obj["type_ids"] = query_token_type_ids+docA_token_type_ids+docB_token_type_ids
        return samples
    
class MonoBERT_Numericalise_Transform():
    def __init__(self, vocab_txt_file="big_files/monoBERT/vocab.txt", **kwargs):
        self.numericalizer = BertTokenizer(vocab_txt_file)
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'query':"text and more", 'doc':"doc text" ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'type_ids':[0,0,1,1], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            query_text = sample_obj['query']
            query_ids = [self.numericalizer.cls_token_id] + self.numericalizer.encode(query_text, add_special_tokens=False)[:62] + [self.numericalizer.sep_token_id]
            query_token_type_ids = [0]*len(query_ids)
            
            doc_text = sample_obj['doc']
            doc_ids = self.numericalizer.encode(doc_text, add_special_tokens=False)[:445] + [self.numericalizer.sep_token_id]
            doc_token_type_ids = [1]*len(doc_ids)
            
            sample_obj["input_ids"] = query_ids+doc_ids
            sample_obj["type_ids"] = query_token_type_ids+doc_token_type_ids
        return samples
    
class Numericalise_Transform():
    def __init__(self, numericaliser='BART', fields=[("input_text","input_ids")], debug=True, max_len=1000, **kwargs):
        if numericaliser == 'BART':
            self.numericaliser = BartTokenizer.from_pretrained('facebook/bart-large').encode
        elif numericaliser == 'T5':
            self.numericaliser = BertTokenizer.from_pretrained('bert-base-uncased').encode
        else:
            self.numericaliser = numericaliser
        if debug:
            print(f"Numericaliser. Ex: 'This is a test' -> {self.numericaliser('This is a test')}")
        self.fields = fields
        self.max_len = max_len
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_text':"text and more", ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.numericaliser(sample_obj[str_field])[:self.max_len]
        return samples