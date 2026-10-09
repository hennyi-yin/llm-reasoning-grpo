import pytest

from src.benchmark_common import digest
from src.quantize import calibration_records


def test_tokenizer_mapping_calibration_provenance_keeps_exact_token_lists():
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

        def __call__(self, text, **kwargs):
            return transformers.BatchEncoding({"input_ids": [7, 11, 13],
                                               "attention_mask": [1, 1, 1]})

    records = calibration_records(Tokenizer(), [{"question": "What is 2 + 2?"}], 1024)
    dataset = datasets.Dataset.from_list(records)
    assert records == [{"input_ids": [7, 11, 13], "attention_mask": [1, 1, 1]}]
    assert digest(records) == digest(dataset.to_list())
