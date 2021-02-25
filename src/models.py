from transformers import BartModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule
import pl_bolts
import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import re

from src.utils import chunks

class PassThroughReWriter():
    def __init__(self, args):
        "simply use the raw query."
        pass
    def inference(self, samples, **kwargs):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_raw_queries'][-1]
        return samples
    
class OracleReWriter():
    def __init__(self, args):
        "simply use the raw query."
        pass
    def inference(self, samples, **kwargs):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_manual_queries'][-1]
        return samples
    
class Transformer_Plus_Plus_Q_QuReTeC_Doc_Context():
    def __init__(self, args):
        "Transformer_Plus_Plus_Q_QuReTeC model from Vakulenko et al."
        pass
    def inference(self, samples, **kwargs):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['6_Transformer_Plus_Plus_Q_QuReTeC_QnA.tsv']
        return samples

class BART_ReWriter(LightningModule):
    def __init__(self, args):
        """
        R1 = Raw 1
        
        Training:
        R1 + R2 + R3 -> M3
        """
        super().__init__()
        self.lr = getattr(args, "lr")
        self.transformer = BartForConditionalGeneration.from_pretrained('facebook/bart-large', force_bos_token_to_be_generated=True)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        
    def forward(self, encoder_input, decoder_input):
        outputs = self.transformer(input_ids, decoder_input_ids=decoder_input)
        return outputs
    
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        input_text = [' '.join(sample['all_raw_queries']) for sample in input_samples]
        target_text = [sample['all_manual_queries'][-1] for sample in input_samples]
        
        input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
        collated_samples['input_ids'] = input_tok_obj['input_ids']
        collated_samples['input_att_mask'] = input_tok_obj['attention_mask']
        
        input_tok_obj = self.tokenizer(target_text, return_tensors='pt', padding=True)
        collated_samples['decoder_input_ids'] = input_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = input_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = input_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_att_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        decoder_mask = batch['decoder_att_mask']
                
        outputs = self.transformer(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.transformer.config.vocab_size), decoder_target.view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=3, max_epochs=10)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    def inference(self, input_samples, max_len=200, chunk_size=64):
        """
        input_samples: [{'all_raw_queries':['sadfad','adfad'], ...}]
        """
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="Re-writing"):
                input_text = [' '.join(sample['all_raw_queries']) for sample in chunk_samples]
                input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
                input_ids = input_tok_obj['input_ids'].to(self.device)
                input_att_mask = input_tok_obj['attention_mask'].to(self.device)

                output_ids = self.transformer.generate(input_ids, attention_mask=input_att_mask, num_beams=4, max_length=max_len, early_stopping=True)
                output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for sample, out_text in zip(chunk_samples, output_text):
                    sample['re-write'] = out_text
                    sample['model output'] = out_text
                new_samples += chunk_samples
            return new_samples
        
class BART_ReWriter_Order_Switch(BART_ReWriter):
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        input_text = [' '.join(sample['all_raw_queries']) for sample in input_samples]
        target_text = [sample['all_manual_queries'][-1] for sample in input_samples]
        
        input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
        collated_samples['input_ids'] = input_tok_obj['input_ids']
        collated_samples['input_att_mask'] = input_tok_obj['attention_mask']
        
        input_tok_obj = self.tokenizer(target_text, return_tensors='pt', padding=True)
        collated_samples['decoder_input_ids'] = input_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = input_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = input_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
        
class BART_All_Queries_ReWriter(BART_ReWriter):
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        batch_size = len(input_samples)
        input_text = [' '.join(sample['all_raw_queries']) for sample in input_samples]
        target_text = [' '.join(sample['all_manual_queries']) for sample in input_samples]
        
        decoder_context = [' '.join(sample['all_manual_queries'][:-1]) for sample in input_samples]
        
        input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
        collated_samples['input_ids'] = input_tok_obj['input_ids']
        collated_samples['input_att_mask'] = input_tok_obj['attention_mask']
        
        input_tok_obj = self.tokenizer(target_text, return_tensors='pt', padding=True)
        decoder_input_ids = input_tok_obj['input_ids'][:,:-1]
        decoder_target_ids = input_tok_obj['input_ids'][:,1:]
        decoder_att_mask = input_tok_obj['attention_mask'][:,1:]
        
        decoder_context_tok_obj = self.tokenizer(decoder_context, return_tensors='pt', padding=True, add_special_tokens=False)
        decoder_context_att_mask = decoder_context_tok_obj['attention_mask']
        length_difference = decoder_att_mask.shape[1] - decoder_context_att_mask.shape[1]
        eq_decoder_context_att_mask = torch.cat([decoder_context_att_mask, torch.zeros((batch_size,length_difference), dtype=torch.long)], dim=1)
        
        grad_mask = (decoder_att_mask & ~eq_decoder_context_att_mask).to(torch.bool)
        
        collated_samples['decoder_input_ids'] = decoder_input_ids
        collated_samples['decoder_target_ids'] = decoder_target_ids
        collated_samples['decoder_att_mask'] = decoder_att_mask
        collated_samples['grad_mask'] = grad_mask
        
        return collated_samples
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_att_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        decoder_mask = batch['decoder_att_mask']
        grad_mask = batch['grad_mask']
                
        outputs = self.transformer(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits[grad_mask].view(-1, self.transformer.config.vocab_size), decoder_target[grad_mask].view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits, 'log': {'loss': {'train': loss}}}
    
    def inference(self, input_samples, max_len=200, chunk_size=64):
        """
        input_samples: [{'all_raw_queries':['sadfad','adfad'], ...}]
        """
        self.eval()
        with torch.no_grad():
            new_samples = []
            for chunk_samples in tqdm(list(chunks(input_samples, chunk_size)), desc="Re-writing"):
                input_text = [' '.join(sample['all_raw_queries']) for sample in chunk_samples]
                input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
                input_ids = input_tok_obj['input_ids'].to(self.device)
                input_att_mask = input_tok_obj['attention_mask'].to(self.device)

                top_outputs = self.transformer.generate(input_ids, attention_mask=input_att_mask, num_beams=4, num_return_sequences=4, max_length=max_len, early_stopping=True)
                chunk_len = len(chunk_samples)
#                 print(top_outputs.reshape(chunk_len, 4,-1).shape)
                output_text = self.tokenizer.batch_decode(top_outputs, skip_special_tokens=True)
#                 print(output_text)
                for sample, out_text in zip(chunk_samples, output_text):
                    all_rewritten_queries = re.findall('[^.?,]+.?', out_text)
                    sample['model output'] = out_text
                    sample['re-write'] = all_rewritten_queries[-1] if len(all_rewritten_queries[-1])>20 else ' '.join(all_rewritten_queries[:-2])
                new_samples += chunk_samples
            return new_samples
        
class BART_ReWriter_BERT_Weights(BART_ReWriter):
    def pre_train_process_transform(self, samples, BERT_dir='big_files/monoBERT'):
        BERT_reranker = BertForPassageRanking.from_pretrained(BERT_dir, from_tf=True)
        BERT_tokenizer = BertTokenizer('big_files/duoBERT/vocab.txt')
        get_doc_fn = CAsT_Index_store().get_doc
        
        for sample in samples:
            query = sample['all_manual_queries'][-1]
            doc = get_doc_fn
            if len(BERT_tokenizer.tokenize(query) != len(self.tokenizer.tokenize(query))):
                uniform_val = 1/len(self.tokenizer.tokenize(query))
                sample['query_token_weights'] = [uniform_val]*len(self.tokenizer.tokenize(query))
            else:
                input_obj = BERT_tokenizer(query, text_pair=doc)
                input_ids = input_obj['input_ids']
                token_type_ids = input_obj['token_type_ids']
                attention_mask = input_obj['attention_mask']
                outputs = BERT_reranker(input_obj['input_ids'], attention_mask=input_obj['attention_mask'], token_type_ids=input_obj['token_type_ids'], output_attentions=True)
        
    def training_step(self, batch, batch_idx):
        encoder_input = batch["input_ids"]
        input_mask = batch['input_att_mask']
        
        decoder_input = batch['decoder_input_ids']
        decoder_target = batch['decoder_target_ids']
        decoder_mask = batch['decoder_att_mask']
                
        outputs = self.transformer(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.transformer.config.vocab_size), decoder_target.view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits}

        
class T5_ReWriter(BART_ReWriter):
    def __init__(self, args):
        super().__init__(args)
        self.transformer = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
