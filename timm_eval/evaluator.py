import itertools
import logging
import random
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
import timm
from tqdm import tqdm

import timm_eval.api.registry
from timm_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
)
from timm_eval.loggers import EvaluationTracker
from timm_eval.loggers.utils import (
    add_env_info,
    get_git_commit_hash,
)
from timm_eval.tasks import (
    TaskManager,
    get_task_dict,
)
from timm_eval.utils import (
    eval_logger,
    positional_deprecated,
    simple_parse_args_string,
    create_iterator,
    compile_and_warmup,
)


if TYPE_CHECKING:
    from timm_eval.api.model import Model
    from timm_eval.api.task import Task

torch.set_float32_matmul_precision("high")


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    compile_model: bool = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, Model]
        Name of model or Model object, see timm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see Model.create_from_arg_string and Model.create_from_arg_object.
        Ignored if `model` argument is a Model object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param compile_model: bool
        Compile and warm up the model before executing tasks.

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    start_date = time.time()

    seed_message = []
    if random_seed is not None:
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""

        if isinstance(model_args, dict):
            eval_logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            model = timm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            eval_logger.info(
                f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
            )
            model = timm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, timm_eval.api.model.Model):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type {type(model)}, but is required to be a subclass of timm_eval.api.model.Model. This may be because you are passing an initialized Hugging Face PreTrainedModel without having wrapped it in `timm_eval.models.huggingface.HFModel(pretrained=my_model)` first."
            )
        eval_logger.info("Using pre-initialized model")

    eval_logger.info("Model has config card " + str(model.config))
    eval_logger.info("Number of patches: " + str(model.model.patch_embed.num_patches))

    if compile_model:
        eval_logger.info(
            "Compiling model with torch.compile and warming up on synthetic input."
        )
        compile_and_warmup(
            model,
            input_size=model.config["input_size"],
            batch_size=int(batch_size),
        )
        # model._model = compile_and_warmup(
        #    model.model,
        #    input_size=model.config["input_size"],
        #    batch_size=int(batch_size),
        # )
        eval_logger.info("Finished compilation and warmup.")

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    task_dict = get_task_dict(tasks, task_manager)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
        )

    results = evaluate(
        model=model,
        task_dict=task_dict,
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        verbosity=verbosity,
    )

    if model.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(model.batch_sizes.values())
                    if hasattr(model, "batch_sizes")
                    else []
                ),
                "device": device,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    model: "Model",
    task_dict: dict,
    batch_size: int,
    limit: Optional[int] = None,
    bootstrap_iters: Optional[int] = 100000,
    verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :return
        Dictionary of results
    """

    RANK = model.rank
    WORLD_SIZE = model.world_size

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    eval_logger.debug(f"Starting evaluate(); model config is {model.config}")
    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    for task_output in eval_tasks:
        task: Task = task_output.task
        limit = get_sample_size(task, limit)
        task.setup_dataset(model.transform)
        dataloader = create_iterator(
            timm.data.loader.create_loader(
                task.eval_imgs,
                model.config["input_size"],
                int(batch_size),
            ),
            rank=RANK,
            limit=limit,
            world_size=WORLD_SIZE,
        )
        eval_logger.debug(
            f"Running task: {task_output.task_name}; number of imgs on this rank: {len(dataloader)}"
        )

        eval_logger.debug(f"Model is on device {model.device}")

        # Reset peak memory tracking for this task
        task_peak_memory = 0
        if (
            hasattr(model, "device")
            and torch.cuda.is_available()
            and "cuda" in str(model.device)
        ):
            device_id = model.device.index if model.device.index is not None else 0
            base_memory = torch.cuda.memory_allocated(device_id)
            torch.cuda.reset_peak_memory_stats(device_id)

        forward_times = []
        # run requests through model
        t_start = time.perf_counter()
        for imgs, targets in tqdm(dataloader):
            fwd_start = time.perf_counter()
            with torch.no_grad():
                outputs = getattr(model, task.OUTPUT_TYPE)(imgs)
            forward_times.append(time.perf_counter() - fwd_start)

            # calculate metrics
            for output, target in zip(outputs, targets):
                metrics = task.process_results(output, target)
                for metric, value in metrics.items():
                    task_output.sample_metrics[metric].append(value)

            # Update peak memory usage for this task
            if (
                hasattr(model, "device")
                and torch.cuda.is_available()
                and "cuda" in str(model.device)
            ):
                device_id = model.device.index if model.device.index is not None else 0
                task_peak_memory = max(
                    task_peak_memory, torch.cuda.max_memory_allocated(device_id)
                )

        task_time = time.perf_counter() - t_start
        min_fwd = min(forward_times)
        median_fwd = np.median(forward_times)
        max_fwd = max(forward_times)
        eval_logger.info(
            f"Task {task_output.task_name} took {task_time:.2f}s. Forward pass times - min: {min_fwd:.6f}s, median: {median_fwd:.6f}s, max: {max_fwd:.6f}s"
        )

        # Log peak memory usage for this task
        if (
            hasattr(model, "device")
            and torch.cuda.is_available()
            and "cuda" in str(model.device)
        ):
            task_peak_memory_gb = task_peak_memory / (1024**3)
            base_memory_gb = base_memory / (1024**3)
            total_peak_memory_gb = (base_memory + task_peak_memory) / (1024**3)
            eval_logger.info(
                f"Task {task_output.task_name} VRAM usage on device {model.device}: {total_peak_memory_gb:.2f} GB total (base: {base_memory_gb:.2f} GB + task peak: {task_peak_memory_gb:.2f} GB)"
            )

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            configs,
            versions,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if (
                len(task_list) != 0
            ):  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            eval_logger.warning(
                                f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
        }

        return results_dict

    else:
        return None


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
