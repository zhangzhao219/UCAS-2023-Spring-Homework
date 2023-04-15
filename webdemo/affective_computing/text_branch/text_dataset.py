import torch
import pandas as pd
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class IMDB_climate_waimai_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, args=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = None
        self.labels = None
        self.read_file()

    def read_file(self):
        """从文件中读入"""
        max_seq_length = 512

        data_file = pd.read_csv(self.file_path)
        text = data_file['text'].values.tolist()  # List[str]
        labels = data_file['label'].values.tolist()

        self.labels = labels
        self.data = self.tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_length)
        
        logger.info('data: ' + str(len(self.data['input_ids'])))
        # print(f'data: {len(self.data['input_ids'])}')

    def __len__(self):
        assert len(self.data['input_ids']) == len(self.labels)
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        # self.data   input_ids, token_type_ids, attention_mask
        # 单个句子的任务可以不设置token_type_ids， 全0
        # item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        item = {}
        item['input_ids'] = torch.tensor(self.data['input_ids'][idx], dtype=torch.long)
        item['attention_mask'] = torch.tensor(self.data['attention_mask'][idx], dtype=torch.long)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item