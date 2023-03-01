import numpy as np
from typing import List, Union

from transformers.utils import logging
from transformers.pipelines.base import ChunkPipeline, ArgumentHandler
from transformers.tokenization_utils import TruncationStrategy
from transformers.pipelines import ZeroShotClassificationArgumentHandler

logger = logging.get_logger(__name__)


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template, reverse):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError(
                "You must include at least one label and at least one sequence."
            )
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        sequence_pairs = []
        for sequence in sequences:
            if reverse:
                sequence_pairs.extend(
                    [[hypothesis_template.format(label), sequence] for label in labels]
                )
            else:
                sequence_pairs.extend(
                    [[sequence, hypothesis_template.format(label)] for label in labels]
                )

        return sequence_pairs, sequences


class NSPZeroShotClassificationPipeline(ChunkPipeline):
    def __init__(
        self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs
    ):
        self._args_parser = args_parser
        super().__init__(*args, **kwargs)

    @property
    def isNext_id(self):
        return 0

    def _parse_and_tokenize(
        self,
        sequence_pairs,
        padding=True,
        add_special_tokens=True,
        truncation=TruncationStrategy.ONLY_FIRST,
        **kwargs,
    ):
        return_tensors = self.framework
        if self.tokenizer.pad_token is None:
            logger.error(
                "Tokenizer was not supporting padding necessary for zero-shot, attempting to use "
                " `pad_token=eos_token`"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                sequence_pairs,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )
        except Exception as e:
            if "too short" in str(e):
                inputs = self.tokenizer(
                    sequence_pairs,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                )
            else:
                raise e

        return inputs

    def _sanitize_parameters(self, **kwargs):
        if kwargs.get("multi_class", None) is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(
                kwargs["candidate_labels"]
            )
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]
        if "reverse" in kwargs:
            preprocess_params["reverse"] = kwargs["reverse"]

        postprocess_params = {}
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        sequences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        if len(args) == 0:
            pass
        elif len(args) == 1 and "candidate_labels" not in kwargs:
            kwargs["candidate_labels"] = args[0]
        else:
            raise ValueError(f"Unable to understand extra arguments {args}")

        return super().__call__(sequences, **kwargs)

    def preprocess(
        self,
        inputs,
        candidate_labels=None,
        hypothesis_template="This example is {}.",
        reverse=False,
    ):
        sequence_pairs, sequences = self._args_parser(
            inputs, candidate_labels, hypothesis_template, reverse
        )

        for i, (candidate_label, sequence_pair) in enumerate(
            zip(candidate_labels, sequence_pairs)
        ):
            model_input = self._parse_and_tokenize([sequence_pair])

            yield {
                "candidate_label": candidate_label,
                "sequence": sequences[0],
                "is_last": i == len(candidate_labels) - 1,
                **model_input,
            }

    def _forward(self, inputs):
        candidate_label = inputs["candidate_label"]
        sequence = inputs["sequence"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        outputs = self.model(**model_inputs)

        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            **outputs,
        }
        return model_outputs

    def postprocess(self, model_outputs, multi_label=False):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        sequences = [outputs["sequence"] for outputs in model_outputs]
        logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
        N = logits.shape[0]
        n = len(candidate_labels)
        num_sequences = N // n
        reshaped_outputs = logits.reshape((num_sequences, n, -1))

        if multi_label or len(candidate_labels) == 1:
            isNext_id = self.isNext_id
            notNext_id = 1
            isNext_contr_logits = reshaped_outputs[..., [notNext_id, isNext_id]]
            scores = np.exp(isNext_contr_logits) / np.exp(isNext_contr_logits).sum(
                -1, keepdims=True
            )
            scores = scores[..., 1]
        else:
            isNext_logits = reshaped_outputs[..., self.isNext_id]
            scores = np.exp(isNext_logits) / np.exp(isNext_logits).sum(
                -1, keepdims=True
            )

        top_inds = list(reversed(scores[0].argsort()))
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist(),
        }
