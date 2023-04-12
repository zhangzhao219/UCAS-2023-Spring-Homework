import torch
import pandas as pd
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class IMDB_climate_waimai_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, args):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.args = args
        self.data = None
        self.labels = None
        self.read_file()

    def read_file(self):
        """从文件中读入"""

        max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)

        data_file = pd.read_csv(self.file_path)
        text = data_file['text'].values.tolist()  # List[str]
        labels = data_file['label'].values.tolist()
        if self.args.dataset_name == 'climate':
            # 标签映射 从0开始, 其实全+1就行
            # set_labels = sorted(list(set(labels)))
            labels = list(map(lambda x:x+1, labels))
            new_text = [t if type(t) == str else "." for t in text]
            text = new_text

        self.labels = labels
        # 进行debug，采样
        if self.args.max_debug_samples != 0:
            text = text[:self.args.max_debug_samples]
            self.labels = self.labels[:self.args.max_debug_samples]
            
        # tokenizer得到的是input_ids attention mask等内容，传入必须是str, List[str], List[List[str]]
        # # True或”longest“：填充到最长序列（如果你仅提供单个序列，则不会填充）；
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