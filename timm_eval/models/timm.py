from datetime import timedelta
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import timm
import numpy as np
from timm.layers.pos_embed import resample_abs_pos_embed
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from huggingface_hub import HfApi
from packaging import version

from timm_eval import utils
from timm_eval.api.model import Model
from timm_eval.api.registry import register_model


eval_logger = utils.eval_logger


@register_model("timm")
class TIMMModel(Model):
    def __init__(
        self,
        pretrained: str,
        flex_patch_embed: Optional[bool] = False,
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))
        self.pretrained = pretrained

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        ps = None
        if flex_patch_embed:
            ps = kwargs.pop("patch_size", None)
            if not ps:
                raise ValueError(
                    f"Got {flex_patch_embed=} but also found patch_size {ps}"
                )
        config_updates = kwargs.pop("config", None)
        if config_updates and "input_size" in config_updates:
            input_size = config_updates["input_size"]
            kwargs["img_size"] = (
                input_size[-1] if isinstance(input_size, (list, tuple)) else input_size
            )
        eval_logger.info(f"Creating model with {kwargs}")
        self._create_model(**kwargs)
        self._get_config()
        if config_updates:
            eval_logger.info(f"Updating config with {config_updates}")
            self.update_config(config_updates)

        self._get_transform()
        if ps:
            self._repatch(ps, **kwargs)

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            if hasattr(self.model, "eval") and callable(self.model.eval):
                self.model.eval()
            else:
                eval_logger.info(
                    "Model had no eval(). Make sure model is still in eval mode as desired."
                )
            if hasattr(self.model, "tie_weights") and callable(self.model.tie_weights):
                self.model.tie_weights()
            else:
                eval_logger.info("Model had no tie_weights().")

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        # multigpu data-parallel support when launched with accelerate
        if gpus > 1:
            if accelerator.num_processes > 1:
                if gpus > accelerator.num_processes:
                    eval_logger.warning(
                        "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                        "If you would like to use data parallelism, please launch the script "
                        "with 'accelerate launch *script*'. "
                        f"Current run will proceed with {accelerator.num_processes} devices."
                    )
                    if self.accelerator.is_local_main_process:
                        eval_logger.info(f"Using {gpus} devices with data parallelism")

                self._device = torch.device(f"{accelerator.device}")
                self.accelerator = accelerator

                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                # if we aren't launching via accelerate, ditch
                self._rank = 0
                self._world_size = 1

    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        return self._model_call(batch)

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def transform(self):
        return self._transform

    def _get_config(
        self,
    ) -> None:
        self._config = self._model.pretrained_cfg.copy()

    def update_config(self, config_updates: dict) -> None:
        """Update the model's config with new values.

        Args:
            config_updates: Dictionary of config keys and values to update
        """
        self._config.update(config_updates)

    def _get_transform(self) -> None:
        data_cfg = timm.data.resolve_data_config(self.config)
        self._transform = timm.data.create_transform(**data_cfg)

    def _create_model(self, pretrained: bool = True, **model_kwargs) -> None:
        self._model = timm.create_model(
            self.pretrained, pretrained=pretrained, **model_kwargs
        )
        self._model = self._model.to(self.device)

    def _repatch(self, patch_size: int, **model_kwargs) -> None:
        eval_logger.info(
            f"Repatching model to patch size {patch_size} using pi_resize_patch_embed"
        )
        new_patch_size = (patch_size, patch_size)
        state_dict = self._model.state_dict()
        state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
            patch_embed=state_dict["patch_embed.proj.weight"],
            new_patch_size=new_patch_size,
        )

        # Interpolate the position embedding size
        assert isinstance(self.config["input_size"], (list, tuple))
        image_size = self.config["input_size"][-2:]
        grid_size = [i // p for i, p in zip(image_size, new_patch_size)]
        eval_logger.debug(
            f"{image_size=}, {grid_size=}, {self._model.num_prefix_tokens=}, {state_dict['pos_embed'].shape=}"
        )
        state_dict["pos_embed"] = resample_abs_pos_embed(
            posemb=state_dict["pos_embed"],
            new_size=grid_size,
            num_prefix_tokens=0
            if self._model.no_embed_class
            else self._model.num_prefix_tokens,
            verbose=True,
        )
        ## Load the new weights into a model with the target image and patch sizes
        self._create_model(
            pretrained=False,
            img_size=image_size,
            patch_size=new_patch_size,
            **model_kwargs,
        )
        self._model.load_state_dict(state_dict, strict=True)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)  # in TIMM, this returns the logits

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained)
                return model_info.sha
            except Exception as e:
                eval_logger.warn(
                    f"Failed to get model SHA for {pretrained}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_sha": get_model_sha(self.pretrained),
        }
        return model_info


def pi_resize_patch_embed(
    patch_embed: torch.Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """
    Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized patch_embed of size [d, c, h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    d, c, h, w = patch_embed.shape
    old_patch_size = (h, w)

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: torch.Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape, device=patch_embed.device)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)

    # Reshape patch_embed to [d*c, h*w]
    patch_embed_flat = patch_embed.reshape(d * c, -1)

    # Apply pseudo-inverse resizing
    resized_patch_embed_flat = resize_matrix_pinv @ patch_embed_flat.t()

    # Reshape back to [d, c, h', w']
    resized_patch_embed = resized_patch_embed_flat.t().reshape(d, c, *new_patch_size)

    return resized_patch_embed
