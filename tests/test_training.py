"""CPU checks for experimental parity and actual TRL completion-only labels."""
import copy

import pytest

from src.common import load_config
from src.train_sft import to_sft
from src.train_grpo import to_rl
from src.training import assert_grpo_pair, validate_lora


def test_sft_targets_strip_calculator_annotations_and_use_shared_prompt():
    row = to_sft({"question": "What is 6 / 2?", "answer": "6 / 2 = <<6/2=3>>3.\n#### 3"})
    assert row["prompt"][0]["role"] == "user"
    assert row["completion"] == [{"role": "assistant", "content": "6 / 2 = 3.\nThe final answer is \\boxed{3}."}]
    assert to_rl({"question": "What is 6 / 2?", "answer": "x\n#### 3"})["prompt"] == row["prompt"]


def test_main_and_ablation_have_identical_training_parameters():
    assert_grpo_pair()
    main, ablation = load_config("configs/grpo.yaml"), load_config("configs/grpo_no_sft.yaml")
    assert main["trainer"] == ablation["trainer"]
    assert main["lora"] == ablation["lora"]
    assert main["trainer"]["generation_batch_size"] == 32
    assert main["trainer"]["num_generations"] == 8
    assert main["trainer"]["beta"] == 0
    assert main["trainer"]["loss_type"] == "dapo"


def test_protocol_rejects_changed_lora_rank():
    lora = copy.deepcopy(load_config("configs/sft.yaml")["lora"])
    lora["r"] = 8
    with pytest.raises(ValueError, match="LoRA r"):
        validate_lora(lora)


def test_resumed_peft_adam_update_matches_uninterrupted_bf16_update(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from types import SimpleNamespace
    from src.train_grpo import resume_safe_trainer

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    def new_model():
        return peft.get_peft_model(Tiny().to(torch.bfloat16),
                                  peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))

    def parameters(model):
        return [p for p in model.parameters() if p.requires_grad]

    def step(model, optimizer):
        for parameter in parameters(model):
            parameter.grad = torch.full_like(parameter, .1)
        optimizer.step()
        optimizer.zero_grad()

    uninterrupted = new_model().to(torch.bfloat16)
    first_optimizer = torch.optim.AdamW(parameters(uninterrupted), lr=.01, foreach=True)
    step(uninterrupted, first_optimizer)
    uninterrupted.save_pretrained(tmp_path)
    torch.save(first_optimizer.state_dict(), tmp_path / "optimizer.pt")

    resumed = new_model()
    resumed.load_adapter(tmp_path, "default", is_trainable=True)
    assert {p.dtype for p in parameters(resumed)} == {torch.float32}
    restored_optimizer = torch.optim.AdamW(parameters(resumed), lr=.01, foreach=True)

    class Loader:
        def __init__(self):
            self.model = resumed
            self.args = SimpleNamespace(use_transformers_continuous_batching=True, bf16=True, fp16=False)

        def _load_optimizer_and_scheduler(self, checkpoint):
            restored_optimizer.load_state_dict(
                torch.load(tmp_path / "optimizer.pt", map_location="cpu", weights_only=True))

    resume_safe_trainer(Loader)()._load_optimizer_and_scheduler(str(tmp_path))
    assert {state["exp_avg"].dtype for state in restored_optimizer.state.values()} == {torch.bfloat16}
    step(resumed, restored_optimizer)
    step(uninterrupted, first_optimizer)
    for actual, expected in zip(parameters(resumed), parameters(uninterrupted)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_actual_trl_dataset_masks_every_prompt_token(tmp_path):
    # CI needs only extraction/reward dependencies. The full training environment
    # additionally runs this local, offline integration check without a CUDA context.
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer

    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2, "user": 3, "assistant": 4, "Q": 5, "R": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                       eos_token="<eos>", pad_token="<pad>")
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
                               "{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}")
    row = to_sft({"question": "Q", "answer": "R\n#### 3"})
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                        max_position_embeddings=256, pad_token_id=2, eos_token_id=1))
    args = SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
                     gradient_checkpointing=False, completion_only_loss=True, max_length=128,
                     report_to=[], loss_type="nll")
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                         train_dataset=Dataset.from_list([row]))
    prepared = trainer.train_dataset[0]
    prompt_ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                               return_dict=False)
    assert prepared["input_ids"][:len(prompt_ids)] == prompt_ids
    assert prepared["labels"][:len(prompt_ids)] == [-100] * len(prompt_ids)
    assert prepared["labels"][len(prompt_ids):] == prepared["input_ids"][len(prompt_ids):]
    assert len(prepared["labels"]) > len(prompt_ids)
    short = {"input_ids": [3, 4, 6], "labels": [-100, -100, 6]}
    batch = trainer.data_collator([prepared, short])
    assert batch["labels"][1, :3].tolist() == [-100, -100, 6]
    assert (batch["labels"][1, 3:] == -100).all().item()
