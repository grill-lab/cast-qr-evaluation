from transformers import BartModel, BertTokenizer, BartForConditionalGeneration
from pytorch_lightning.core.lightning import LightningModule

import torch
from torch.nn import functional as F
from torch import nn


class PassThroughReWriter():
    def __init__(self):
        "simply use the raw query."
        pass
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_raw_queries'][-1]
        return samples
    
class OracleReWriter():
    def __init__(self):
        "simply use the raw query."
        pass
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_manual_queries'][-1]
            sample_obj["raw query"] = sample_obj['all_raw_queries'][-1]
        return samples
    
class BART_ReWriter(LightningModule):
    def __init__(self):
        """
        R1 = Raw 1
        R1 + R2 + R3 -> Res3
        """
        self.BART = BartForConditionalGeneration.from_pretrained('facebook/bart-large')