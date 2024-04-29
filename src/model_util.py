import torch
from pytorch_lightning.callbacks import EarlyStopping
from tape.models.modeling_bert import ProteinBertConfig, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP, ValuePredictionHead
from torch import nn
from torch.nn.utils import weight_norm

from src.util import resize_token_embeddings


class SimpleMLPFix(SimpleMLP):
    def __init__(
        self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__(in_dim, hid_dim, out_dim, dropout)
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        )


class ValuePredictionHeadFix(ValuePredictionHead):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.hidden_dropout_prob)
        self.value_prediction = SimpleMLPFix(
            config.hidden_size,
            int(config.hidden_size * 2 / 3),
            1,
            config.hidden_dropout_prob,
        )


ValuePredictionHead = ValuePredictionHeadFix


class CCSValuePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.value_prediction = SimpleMLPFix(
            config.hidden_size + 4,
            int(config.hidden_size * 2 / 3),
            1,
            config.hidden_dropout_prob,
        )

    def forward(self, pooled_output, charge):
        value_pred = self.value_prediction(
            torch.cat([pooled_output, charge], dim=1)
        )

        return value_pred


class EarlyStoppingLate(EarlyStopping):
    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
            or trainer.current_epoch <= trainer.min_epochs
        )


def create_model(args):
    from src.lit_model import LitMTL

    bert_config = ProteinBertConfig.from_pretrained(
        "bert-base",
        vocab_size=len(args.vocab),
        hidden_act=args.activation,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_hidden_layers=args.num_layers,
    )

    if args.mode == "supervised":
        if args.pretrained_model == "own":
            model = LitMTL.load_from_checkpoint(
                args.checkpoint_path,
                strict=False,
                mtl_config=args,
                bert_config=bert_config,
            )

        elif args.pretrained_model == "none":
            model = LitMTL(args, bert_config)

        elif args.pretrained_model == "tape":
            # Use the default config when loading the pretrained model
            bert_config = ProteinBertConfig.from_pretrained("bert-base")
            model = LitMTL(args, bert_config)

            # Set the pretrained TAPE weights with the default config
            model.model.bert = ProteinBertModel.from_pretrained("bert-base")

            # The built-in TAPE ProteinModel resize_token_embeddings function does not work, this is an adapted version
            resize_token_embeddings(model.model, len(args.vocab))

        else:
            raise ValueError(
                f"Using pretrained model '{args.pretrained_model}' not supported"
            )

    elif args.mode == "pretrain":
        model = LitMTL(args, bert_config)

    else:
        raise ValueError(f"Train mode {args.mode} not supported")

    return model
