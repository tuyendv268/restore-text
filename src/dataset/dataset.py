from torch.utils.data import Dataset
from src.utils import prepare_data
import torch

class Dataset(Dataset):
    def __init__(self, input_ids ,label_ids, max_sent_lenth, tokenizer, tag2index):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.max_sent_lenth = max_sent_lenth
        self.tokenizer = tokenizer
        self.tag2index = tag2index

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_id=self.input_ids[index]
        label_id=self.label_ids[index]
        max_sent_length = self.max_sent_lenth

        input_ids, input_masks, label_ids, label_masks = prepare_data(
            input_ids= input_id,
            label_ids= label_id,
            max_sent_length = max_sent_length,
            tokenizer=self.tokenizer,
            tag2index=self.tag2index
        )
    
        input_ids = torch.tensor(input_ids)
        input_masks = torch.tensor(input_masks, dtype=torch.bool)
        label_ids = torch.tensor(label_ids)
        label_masks = torch.tensor(label_masks, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "input_masks": input_masks, 
            "label_ids": label_ids, 
            "label_masks": label_masks
        }
