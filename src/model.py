from tape.models.modeling_bert import (
    ProteinBertAbstractModel,
    ProteinBertModel,
)
from tape.models.modeling_utils import MLMHead
from torch import nn

from src.model_util import CCSValuePredictionHead, ValuePredictionHead


class MTLTransformerEncoder(ProteinBertAbstractModel):
    """
    Multi Task Learning Transformer Encoder based on ProteinBert from TAPE

    Can be pretrained using a Masked Language Model head or can learn properties using multiple task heads
    """

    def __init__(self, bert_config, mode, tasks):
        """

        :param bert_config:         A ProteinBertConfig containing the model parameters
        :param mode:                'supervised' for property prediction or 'pretrain' for MLM pretraining
        :param tasks:               A list of tasks to use for supervised learning, each task will get its own
                                    PredictionHead. CCS prediction gets a slightly different predictionhead including
                                    the Charge of the peptide.
        """
        super().__init__(bert_config)

        self.mode = mode
        self.bert = ProteinBertModel(bert_config)

        if self.mode == "supervised":
            self.task_heads = nn.ModuleDict()
            for t in tasks:
                if t == "CCS":
                    self.task_heads[t] = CCSValuePredictionHead(bert_config)
                else:
                    self.task_heads[t] = ValuePredictionHead(bert_config)
        elif self.mode == "pretrain":
            self.mlm_head = MLMHead(
                bert_config.hidden_size,
                bert_config.vocab_size,
                bert_config.hidden_act,
                ignore_index=0,
            )
        else:
            raise RuntimeError(f"Unrecognized mode {mode}")

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if self.mode == "pretrain":
            self._tie_or_clone_weights(
                self.mlm_head.decoder, self.bert.embeddings.word_embeddings
            )

    def forward(self, input_ids, charge=None, task=None, input_mask=None):
        if self.mode == "supervised":
            if task is None:
                raise RuntimeError("Please specify your prediction task")

            if task == "CCS" and charge is None:
                raise RuntimeError("Charges must be given when task is CCS")

        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]

        if self.mode == "supervised":
            if task == "CCS":
                out = self.task_heads[task](pooled_output, charge)
            else:
                (out,) = self.task_heads[task](pooled_output)
        elif self.mode == "pretrain":
            # add hidden states and attention if they are here
            out = self.mlm_head(sequence_output)[0]

        # (loss), prediction_scores, (hidden_states), (attentions)
        return (out,) + outputs[2:]
