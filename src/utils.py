from contextlib import contextmanager, nullcontext
import json
import os
import torch
import torch.nn as nn
import peft
from peft import PeftModel, LoraConfig, PrefixTuningConfig
from omegaconf import DictConfig, OmegaConf
from internvl_model_wrapper import InternVLModelWrapper
import re

OmegaConf.register_new_resolver("eval", eval, replace=True)


class NullPeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        # MimIC-style experiments optimize only the shift encoder, so keep the
        # backbone frozen to avoid unnecessary gradient storage.
        model.requires_grad_(False)
        self.config = model.config

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def disable_adapter(self):
        return nullcontext()

    def save_pretrained(self, save_directory: str, **kwargs):
        pass

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)

    @property
    def device(self):
        if hasattr(self.base_model, "device"):
            return self.base_model.device
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class ModuleDeviceManager:
    def __init__(self, device):
        self.device = device
        self.old_devices = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, old_device in self.old_devices.items():
            module.to(old_device)

    def move_module(self, module):
        self.old_devices[module] = next(module.parameters()).device
        module.to(self.device)


def get_runtime_device():
    if not torch.cuda.is_available():
        print("Running on CPU.")
        return torch.device("cpu")

    device = torch.device("cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_name = torch.cuda.get_device_name(device)
    if visible:
        print(
            f"Running on GPU: {gpu_name} "
            f"(CUDA_VISIBLE_DEVICES={visible}, using logical cuda:0)"
        )
    else:
        print(f"Running on GPU: {gpu_name}")
    return device


@contextmanager
def disable_irrelevant_awq_probe(model):
    """Skip PEFT's broken AutoAWQ probe for ordinary non-AWQ checkpoints."""
    quantization_method = str(
        getattr(getattr(model, "config", None), "quantization_method", "") or ""
    ).lower()
    if quantization_method == "awq":
        yield
        return

    # PEFT 0.14 eagerly probes an installed AutoAWQ backend for every Linear
    # layer. AutoAWQ in this environment is incompatible with Transformers
    # 4.57, while our QwenVL checkpoint is regular fp16. Disable only that
    # irrelevant backend probe; torch.nn.Linear LoRA dispatch is unchanged.
    from peft.tuners.lora import awq as peft_awq_backend

    original_awq_check = peft_awq_backend.is_auto_awq_available
    peft_awq_backend.is_auto_awq_available = lambda: False
    try:
        yield
    finally:
        peft_awq_backend.is_auto_awq_available = original_awq_check


def convert_to_peft(cfg, lmm):
    peft_config_cls = None
    if cfg.model_name in cfg.peft:
        peft_cfg = OmegaConf.to_container(cfg.peft[cfg.model_name], resolve=True)
        if cfg.peft.name == "lora":
            peft_config_cls = LoraConfig
        elif cfg.peft.name == "prefix-tuning":
            peft_config_cls = PrefixTuningConfig

    if peft_config_cls is None:
        model = NullPeftModel(lmm.model)
    else:
        peft_config = peft_config_cls(**peft_cfg)
        if cfg.peft.name == "lora":
            if isinstance(lmm, InternVLModelWrapper):
                lmm.model.requires_grad_(False)
                with disable_irrelevant_awq_probe(lmm.model.language_model):
                    lmm.model.language_model = peft.get_peft_model(
                        lmm.model.language_model, peft_config
                    )
                return
            with disable_irrelevant_awq_probe(lmm.model):
                model = peft.get_peft_model(lmm.model, peft_config)
        else:
            model = peft.get_peft_model(lmm.model, peft_config)
    lmm.model = model


MODEL_REGISTRY = {
    "qwen2.5-math-7b-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen2.5-math-7b": "Qwen/Qwen2.5-Math-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "internvl2_5-8b": "OpenGVLab/InternVL2_5-8B",
    "llava-onevision-1.5-8b": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "idefics3-8b-llama3": "HuggingFaceM4/Idefics3-8B-Llama3",
}


def _build_model_info(cfg, expected_family=None):
    model_name = str(cfg.model_name).lower()
    if expected_family and expected_family not in model_name:
        raise ValueError(f"Model {cfg.model_name!r} is not a {expected_family} model.")
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model {cfg.model_name!r}. Supported models: {supported}")

    hf_id = MODEL_REGISTRY[model_name]
    model_path = getattr(cfg, "model_path", None) or os.environ.get("LIFT_MODEL_PATH") or hf_id
    print(f"Model source: {model_path} (HF id: {hf_id})")
    return hf_id, model_path


def build_qwen_model_info(cfg):
    return _build_model_info(cfg, "qwen")


def build_llama_model_info(cfg):
    return _build_model_info(cfg, "llama")


def build_internlm_model_info(cfg):
    return _build_model_info(cfg, "internlm")


def build_internvl_model_info(cfg):
    return _build_model_info(cfg, "internvl")


def build_llava_model_info(cfg):
    return _build_model_info(cfg, "llava")


def build_idefics3_model_info(cfg):
    return _build_model_info(cfg, "idefics3")


def build_model(cfg):
    """Return the public model id and a configurable local/Hugging Face source."""
    return _build_model_info(cfg)


def save_pretrained(save_directory, lmm, encoder):
    os.makedirs(save_directory, exist_ok=True),
    encoder_sd = {
        k: v for k, v in encoder.state_dict().items() if not k.startswith("lmm")
    }
    torch.save(encoder_sd, os.path.join(save_directory, "encoder.pth"))
    if (
        isinstance(lmm, InternVLModelWrapper)
        and isinstance(getattr(lmm.model, "language_model", None), PeftModel)
    ):
        lmm.model.language_model.save_pretrained(
            save_directory, save_embedding_layers=False
        )
        return
    lmm.model.save_pretrained(save_directory)


def load_from_pretrained(save_directory, lmm, encoder):
    sd = torch.load(os.path.join(save_directory, "encoder.pth"), weights_only=True)
    if sd:
        missing_keys, unexpected_keys = encoder.load_state_dict(sd, strict=False)

        # keys started with "lmm" is not related to shift encoder
        missing_keys = [k for k in missing_keys if not k.startswith("lmm")]
        if missing_keys:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if os.path.exists(os.path.join(save_directory, "adapter_config.json")):
        if isinstance(lmm, InternVLModelWrapper):
            with disable_irrelevant_awq_probe(lmm.model.language_model):
                lmm.model.language_model = PeftModel.from_pretrained(
                    lmm.model.language_model, save_directory
                )
        else:
            with disable_irrelevant_awq_probe(encoder.lmm.model):
                lmm.model = PeftModel.from_pretrained(encoder.lmm.model, save_directory)


def parse_only_shift_at_layer(value):
    if not isinstance(value, str):
        return value

    try:
        parsed_value = json.loads(value)
        if isinstance(parsed_value, (list, int, type(None))):
            return parsed_value
    except json.JSONDecodeError:
        pass

    stripped = value.strip()
    if stripped.lower() == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return value


# update execute_eval in eval_old.py if runname format is changed
def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train, eval or analyze config.
    """
    if hasattr(cfg, "runname"):
        # training mode
        if cfg.data.num_shot == 0:
            # peft
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        # runname-model-dataset-training_samples-num_shot
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

    if hasattr(cfg, "record_dir"):
        # analyze mode
        # record dir format: path/to/{expand-runname}
        return os.path.basename(cfg.record_dir)

    if getattr(cfg, "ckpt_path", None):
        # eval from a specific checkpoint
        # checkpoint dir format: path/to/{expand-runname}/epoch-{epoch}
        return os.path.basename(os.path.dirname(cfg.ckpt_path))
    else:
        # eval from ICL
        # icl-model-dataset-[num_shot]shot-[cot/direct]
        cot_suffix = "cot" if cfg.use_cot else "direct"
        return f"icl-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_shot}shot-{cot_suffix}"
