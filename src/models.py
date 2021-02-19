from transformers import BartModel, BertTokenizer, BartForConditionalGeneration, BartTokenizer
from pytorch_lightning.core.lightning import LightningModule
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
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_raw_queries'][-1]
        return samples
    
class OracleReWriter():
    def __init__(self, args):
        "simply use the raw query."
        pass
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_manual_queries'][-1]
            sample_obj["raw query"] = sample_obj['all_raw_queries'][-1]
        return samples
    
class BART_ReWriter(LightningModule):
    def __init__(self, args):
        """
        R1 = Raw 1
        
        Training:
        R1 + R2 + R3 -> M3
        """
        super().__init__()
        self.lr = args.lr
        self.BART = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        
    def forward(self, encoder_input, decoder_input):
        outputs = self.BART(input_ids, decoder_input_ids=decoder_input)
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
                
        outputs = self.BART(encoder_input, 
                            decoder_input_ids=decoder_input, 
                            attention_mask=input_mask, 
                            decoder_attention_mask=decoder_mask, 
                            use_cache=False)  
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.BART.config.vocab_size), decoder_target.view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    def inference(self, input_samples, max_len=20, chunk_size=64):
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

                output_ids = self.BART.generate(input_ids, attention_mask=input_att_mask, num_beams=4, max_length=max_len, early_stopping=True)
                output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for sample, out_text in zip(chunk_samples, output_text):
                    sample['re-write'] = out_text
                new_samples += chunk_samples
            return new_samples
        
class BART_All_Queries_ReWriter(BART_ReWriter):
    def collate(self, input_samples):
        """
        input_samples: [dict]: these are samples obtained through the __getitem__ method
        """
        collated_samples = {}
        input_text = [' '.join(sample['all_raw_queries']) for sample in input_samples]
        target_text = [' '.join(sample['all_manual_queries']) for sample in input_samples]
        
        input_tok_obj = self.tokenizer(input_text, return_tensors='pt', padding=True)
        collated_samples['input_ids'] = input_tok_obj['input_ids']
        collated_samples['input_att_mask'] = input_tok_obj['attention_mask']
        
        input_tok_obj = self.tokenizer(target_text, return_tensors='pt', padding=True)
        collated_samples['decoder_input_ids'] = input_tok_obj['input_ids'][:,:-1]
        collated_samples['decoder_target_ids'] = input_tok_obj['input_ids'][:,1:]
        collated_samples['decoder_att_mask'] = input_tok_obj['attention_mask'][:,1:]
        
        return collated_samples
    
    def inference(self, input_samples, max_len=20, chunk_size=64):
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

                output_ids = self.BART.generate(input_ids, attention_mask=input_att_mask, num_beams=4, max_length=max_len, early_stopping=True)
                output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for sample, out_text in zip(chunk_samples, output_text):
                    all_rewritten_queries = re.findall('[^.?,]+.?', out_text)
                    sample['re-write'] = all_rewritten_queries[-1]
                new_samples += chunk_samples
            return new_samples