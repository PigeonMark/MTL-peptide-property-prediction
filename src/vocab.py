import pandas as pd
from tape.tokenizers import IUPAC_VOCAB, TAPETokenizer


def extend_iupac_vocab(modifications):
    new_vocab = IUPAC_VOCAB.copy()
    last_tokeni = IUPAC_VOCAB[next(reversed(IUPAC_VOCAB))]
    for mod_i, toki in zip(
        modifications.values(),
        range(last_tokeni + 1, last_tokeni + 1 + len(modifications)),
    ):
        new_vocab[mod_i] = toki
    return new_vocab


class SeqVocab(TAPETokenizer):
    def __init__(self, modifications):
        self.vocab = extend_iupac_vocab(modifications)
        self.modifications = modifications
        self.tokens = list(self.vocab.keys())
        self._vocab_type = "iupac_extended"
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def pad_i(self) -> int:
        return self.convert_token_to_id(self.pad_token)


def create_vocab(args):
    if args.use_1_data_file:
        df = pd.read_csv(args.data_file, index_col=0)
    else:
        dfs = []
        for f in args.train_file, args.val_file, args.test_file:
            if f is not None:
                dfs.append(pd.read_csv(f, index_col=0))
        df = pd.concat(dfs)

    # Get all unique UNIMOD modifications in the data
    modifications = df["modified_sequence"].str.extractall(
        "([A-Z]?)(\[UNIMOD:[0-9]+\])"
    )
    modifications = modifications.drop_duplicates().reset_index(drop=True)
    modifications = modifications.fillna("_")
    if len(modifications) > 26:
        raise RuntimeError(
            "Currently maximum 26 amino acid-PTM combinations are supported because they are internally "
            "represented as lowercase letters"
        )

    mod2letter = {}
    for i, (aa, mod) in enumerate(zip(modifications[0], modifications[1])):
        aa_mod = aa + mod
        lowercase_letter = chr(97 + i)
        mod2letter[aa_mod] = lowercase_letter

    return SeqVocab(mod2letter)
