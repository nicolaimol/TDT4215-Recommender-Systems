import random
import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self,
                 data_frame,
                 group_column,
                 data_column,
                 train_history=128,
                 valid_history=5,
                 padding_mode="right",
                 split_mode="train",
                 threshold=3.5,
                 threshold_column=None,
                 ):
        self.data_frame = data_frame
        self.group_column = group_column
        self.data_column = data_column
        self.train_history = train_history
        self.valid_history = valid_history
        self.padding_mode = padding_mode
        self.split_mode = split_mode
        self.threshold = threshold
        self.threshold_column = threshold_column
        self.timestamp_column = "time"

        """if threshold_column:
            self.data_frame = self.data_frame[self.data_frame[threshold_column] > threshold]
            self.data_frame = self.data_frame.reset_index(inplace=True)"""

        self.groups_df = self.data_frame.groupby(self.group_column)
        self.groups = list(self.groups_df.groups)

    def pad_sequence(self, tokens, padding_mode="left"):
        if padding_mode == "right":
            return tokens + [0] * (self.train_history - len(tokens))
        elif padding_mode == "left":
            return [0] * (self.train_history - len(tokens)) + tokens
        else:
            raise ValueError("Padding mode must be either 'right' or 'left'")

    def get_sequence(self, group_df):
        if self.split_mode == "train":
            i = group_df.shape[0] - self.valid_history
            end_index = random.randint(10, i if i >= 10 else 10)
        elif self.split_mode in ["valid", "test"]:
            end_index = group_df.shape[0]
        else:
            raise ValueError("Split mode must be either 'train', 'valid' or 'test'")

        start_index = max(0, end_index - self.train_history)

        return group_df[start_index:end_index]

    def mask_sequence(self, sequence, p=0.8):
        return [1 if random.random() > p else token for token in sequence]

    def mask_last_elements_sequence(self, sequence):
        return sequence[:-self.valid_history] + self.mask_sequence(
            sequence[-self.valid_history:], p=0.5
        )

    def get_item(self, idx):

        group = self.groups[idx]

        group_df = self.groups_df.get_group(group)
        group_df = group_df.sort_values(by=[self.timestamp_column])
        group_df.reset_index(inplace=True)

        sequence = self.get_sequence(group_df)

        trg_items = sequence[self.data_column].tolist()

        if self.split_mode == "train":
            src_items = self.mask_sequence(trg_items)
        else:
            src_items = self.mask_last_elements_sequence(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"

        trg_items = self.pad_sequence(trg_items, padding_mode=pad_mode)
        src_items = self.pad_sequence(src_items, padding_mode=pad_mode)

        trg_mask = [1 if token != 0 else 0 for token in trg_items]
        src_mask = [1 if token != 0 else 0 for token in src_items]

        trg_items = torch.tensor(trg_items, dtype=torch.long)
        src_items = torch.tensor(src_items, dtype=torch.long)
        trg_mask = torch.tensor(trg_mask, dtype=torch.long)
        src_mask = torch.tensor(src_mask, dtype=torch.long)

        return {
            "source": src_items,
            "target": trg_items,
            "source_mask": src_mask,
            "target_mask": trg_mask,
        }

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.get_item(idx)