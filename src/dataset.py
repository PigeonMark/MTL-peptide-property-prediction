import random
import re

import torch
from torch.utils.data import Dataset, default_collate

from src.util import end_padding


class MTLPepDataset(Dataset):
    def __init__(self, df, args):
        self.df = df.sample(frac=1)
        self.args = args
        self.pep_col = "modified_sequence"
        self.mask_prob = (
            0.15  # hardcoded for now, could be added as an argument
        )
        self._replace_mods()

    def _replace_mods(self):
        for mod, letter in self.args.vocab.modifications.items():
            if mod.startswith("_"):
                self.df[self.pep_col] = self.df[self.pep_col].str.replace(
                    "^" + re.escape(mod[1:]), letter, regex=True
                )
            else:
                self.df[self.pep_col] = self.df[self.pep_col].str.replace(
                    mod, letter
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if self.args.mode == "supervised":
            return self._getitem_supervised(item)
        else:
            return self._getitem_pretrain(item)

    def _getitem_supervised(self, item):
        peptide = self.df[self.pep_col].iloc[item]
        ids = self.args.vocab.convert_tokens_to_ids(
            peptide[: self.args.seq_len]
        )
        ids = end_padding(ids, self.args.seq_len, self.args.vocab.pad_i)
        task = self.df["task"].iloc[item]
        if task == "CCS":
            charge = self.df["Charge"].iloc[item]
            charge = min(int(charge), 4)
            one_hot = [0.0 for _ in range(4)]
            one_hot[charge - 1] = 1.0
        else:
            one_hot = [0.0 for _ in range(4)]
        label = self.df["label"].iloc[item]
        standardized_label = self.args.scalers[task].transform([[label]])

        return {
            "token_ids": torch.tensor(ids),
            "standardized_label": torch.tensor(standardized_label[0][0]),
            "task": task,
            "charge": torch.tensor(one_hot),
            "indx": self.df.index[item],
        }

    def _getitem_pretrain(self, item):
        peptide = self.df[self.pep_col].iloc[item]
        masked_ids, output_labels = self._mask_token_seq(
            peptide[: self.args.seq_len]
        )
        masked_ids = end_padding(
            masked_ids, self.args.seq_len, self.args.vocab.pad_i
        )
        output_labels = end_padding(
            output_labels, self.args.seq_len, self.args.vocab.pad_i
        )

        return {
            "masked_token_ids": torch.tensor(masked_ids),
            "label": torch.tensor(output_labels),
        }

    def _mask_token_seq(self, token_seq):
        masked_seq = []
        output_label = []

        for token in token_seq:
            prob = random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_seq.append(
                        self.args.vocab.convert_token_to_id(
                            self.args.vocab.mask_token
                        )
                    )

                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_seq.append(random.randrange(len(self.args.vocab)))

                # 10% randomly change token to current token
                else:
                    masked_seq.append(
                        self.args.vocab.convert_token_to_id(token)
                    )
                output_label.append(self.args.vocab.convert_token_to_id(token))
            else:
                masked_seq.append(self.args.vocab.convert_token_to_id(token))
                output_label.append(0)

        return masked_seq, output_label


def custom_collate(data):
    # Use the default collate function for everything except the task, this becomes a list of strings
    coll_data = default_collate(
        [
            {k: v for k, v in d.items() if k not in ["task", "indx"]}
            for d in data
        ]
    )

    if "task" in data[0]:
        coll_data["task"] = [d["task"] for d in data]
    if "indx" in data[0]:
        coll_data["indx"] = [d["indx"] for d in data]

    return coll_data
