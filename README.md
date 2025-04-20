# timm_eval
## Like the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), but for CV

*Note: This is currently in an alpha state. There's a lot of code paths that were inherited from forking lm-evaluation-harness that are completely untested, e.g. using `accelerate`.*

`lm-evaluation-harness` is great. Apparently it doesn't seem like anything like it but for TIMM/PyTorch Image Models already exists online (?), so I hacked this together.

### Installation
Just run `pip install -e .` All dependencies are managed by the `pyproject.toml` and should be accurate (submit a PR if not!). You might want to do this inside of some sort of virtual environment, like the built-in `venv` or `uv`.

### Running
Here's an example running the pre-trained ViT [vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k) on `imagenet1k`.
*NOTE: you must first configure the path to your imagenet folder [here](https://github.com/theoxo/timm_eval/blob/main/timm_eval/tasks/imagenet1k/imagenet1k.yaml). Replace `dataset_path` with the path, e.g. `/path/to/my/imagenet/`.*
```
timm_eval --model timm --model_args pretrained=timm/vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k --tasks imagenet1k --batch_size 512"
```
(The batch size is obviously up to you.)

Some notes on running:
- The `--model` should always be set to `timm`, because this is the only source of model currently supported. I suppose in theory it would not be hard to support loading a model trained locally and so on, I just didn't have use for it at the time.
- `model_args` can configure pretty much anything you can pass to the config of a timm model, I hope.
- I only implemented the imagenet task, but it should be trivial to copy its implementation to other tasks like mnist, cifar, coco, etc.
- Most arguments supported by lm-evaluation-harness` as of summer 2024 should be supported here, too, but most of them are untested (like using accelerate for DDP etc; vision models are currently very small anyway).
