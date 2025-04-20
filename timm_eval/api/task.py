import abc
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from inspect import getsource
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Union,
)

import timm

from timm_eval.api.instance import OutputType, ALL_OUTPUT_TYPES
from timm_eval.api.registry import (
    AGGREGATION_REGISTRY,
    DEFAULT_METRIC_REGISTRY,
    get_aggregation,
    get_metric,
    get_metric_aggregation,
    is_higher_better,
)


eval_logger = logging.getLogger("timm-eval")


@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: Optional[str] = None
    task_alias: Optional[str] = None
    tag: Optional[Union[str, list]] = None
    group: Optional[Union[str, list]] = None
    # dataset options.
    # which dataset to use,
    # and what splits for what purpose
    dataset_path: Optional[str] = None
    dataset_type: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_kwargs: Optional[dict] = None
    training_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None
    # output processing and options
    process_results: Optional[Union[Callable, str]] = None
    metric_list: Optional[list] = None
    output_type: OutputType = "logits"
    metadata: Optional[dict] = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )

    def __post_init__(self) -> None:
        if self.group is not None:
            eval_logger.warning(
                "A task YAML file was found to contain a `group` key. Groups which provide aggregate scores over several subtasks now require a separate config file--if not aggregating, you may want to use the `tag` config option instead within your config. Setting `group` within a TaskConfig will be deprecated in v0.4.4. Please see https://github.com/EleutherAI/timm-evaluation-harness/blob/main/docs/task_guide.md for more information."
            )

            if self.tag is None:
                self.tag = self.group
            else:
                raise ValueError(
                    "Got both a `group` and `tag` entry within a TaskConfig. Please use one or the other--`group` values will be deprecated in v0.4.4."
                )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self, keep_callable: bool = False) -> dict:
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif k == "metric_list":
                for metric_dict in v:
                    for metric_key, metric_value in metric_dict.items():
                        if callable(metric_value):
                            metric_dict[metric_key] = self.serialize_function(
                                metric_value, keep_callable=keep_callable
                            )
                cfg_dict[k] = v
            elif callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Union[Callable, str], keep_callable=False
    ) -> Union[Callable, str]:
        """Serializes a given function or string.

        If 'keep_callable' is True, the original callable is returned.
        Otherwise, attempts to return the source code of the callable using 'getsource'.
        """
        if keep_callable:
            return value
        else:
            try:
                return getsource(value)  # pyright: ignore
            except (TypeError, OSError):
                return str(value)


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See ImageNet1k for a simple example implementation

    An `img` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"image": ..., "label": ...}
    """

    VERSION: Optional[Union[int, str]] = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: Optional[str] = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: Optional[str] = None

    OUTPUT_TYPE: Optional[OutputType] = None

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        config: Optional[Mapping] = None,  # Union[dict, TaskConfig]
    ) -> None:
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        """
        self._training_imgs: Optional[list] = None
        self._config: TaskConfig = TaskConfig(**config) if config else TaskConfig()
        self.setup_dataset_paths()

    @abc.abstractmethod
    def setup_dataset_paths(self):
        pass

    @abc.abstractmethod
    def setup_dataset(self, transform):
        pass

    @property
    def config(self) -> TaskConfig:
        """Returns the TaskConfig associated with this class."""
        return self._config

    @abc.abstractmethod
    def has_training_imgs(self) -> bool:
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_imgs(self) -> bool:
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_imgs(self) -> bool:
        """Whether the task has a test set"""
        pass

    def training_imgs(self) -> Iterable:
        return []

    def validation_imgs(self) -> Iterable:
        return []

    def test_imgs(self) -> Iterable:
        return []

    @abc.abstractmethod
    def process_results(self, output, label):
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abc.abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        pass

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
        return self.config.to_dict()

    def set_config(self, key: str, value: Any, update: bool = False) -> None:
        """Set or update the configuration for a given key."""
        if key is None:
            raise ValueError("Key must be provided.")

        if update:
            current_value = getattr(self._config, key, {})
            if not isinstance(current_value, dict):
                raise TypeError(
                    f"Expected a dict for key '{key}', got {type(current_value).__name__} instead."
                )
            current_value.update(value)
        else:
            setattr(self._config, key, value)

    def override_metric(self, metric_name: str) -> None:
        """
        Override the default metrics used for evaluation with custom metrics.

        Parameters:
        - metric_name (str): The name of the custom metric to override. Should be registered in api.metrics.
        """
        (
            self._metric_fn_list,
            self._aggregation_list,
            self._metric_fn_kwargs,
            self._higher_is_better,
        ) = ({}, {}, {}, {})
        self._metric_fn_list[metric_name] = get_metric(metric_name)
        self._aggregation_list[metric_name] = get_metric_aggregation(metric_name)
        self._higher_is_better[metric_name] = is_higher_better(metric_name)
        self._metric_fn_kwargs[metric_name] = {}
        if not isinstance(self, ConfigurableTask):
            self.process_results = lambda x, y: {metric_name: get_metric(metric_name)}
            self.aggregation = lambda: {
                metric_name: get_metric_aggregation(metric_name)
            }
        setattr(self._config, "metric_list", [{"metric": metric_name}])
        setattr(self._config, "process_results", None)

    @property
    def eval_imgs(
        self,
    ) -> timm.data.dataset.ImageDataset:
        if self.has_test_imgs():
            return self.test_imgs()
        elif self.has_validation_imgs():
            return self.validation_imgs()
        else:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test imgs!"
            )


class ConfigurableTask(Task):
    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config: Optional[dict] = None,
    ) -> None:  # TODO no super() call here
        # Get pre-configured attributes
        self._config = self.CONFIG

        # Use new configurations if there was no preconfiguration
        if self.config is None:
            self._config = TaskConfig(**config)
        # Overwrite configs
        else:
            if config is not None:
                self._config.__dict__.update(config)

        if self.config is None:
            raise ValueError(
                "Must pass a config to ConfigurableTask, either in cls.CONFIG or `config` kwarg"
            )

        if isinstance(self.config.metadata, dict):
            if "version" in self.config.metadata:
                self.VERSION = self.config.metadata["version"]

        if self.config.output_type is not None:
            if self.config.output_type not in ALL_OUTPUT_TYPES:
                raise ValueError(
                    f"Got invalid output_type '{self.config.output_type}', must be in '{','.join(ALL_OUTPUT_TYPES)}'"
                )
            self.OUTPUT_TYPE = self.config.output_type

        if self.config.dataset_path is not None:
            self.DATASET_PATH = self.config.dataset_path

        if self.config.dataset_name is not None:
            self.DATASET_NAME = self.config.dataset_name

        self._metric_fn_list = {}
        self._metric_fn_kwargs = {}
        self._aggregation_list = {}
        self._higher_is_better = {}

        if self.config.metric_list is None:
            # TODO: handle this in TaskConfig.__post_init__ ?
            _metric_list = DEFAULT_METRIC_REGISTRY[self.config.output_type]

            for metric_name in _metric_list:
                self._metric_fn_list[metric_name] = get_metric(metric_name)
                self._metric_fn_kwargs[metric_name] = {}
                self._aggregation_list[metric_name] = get_metric_aggregation(
                    metric_name
                )
                self._higher_is_better[metric_name] = is_higher_better(metric_name)
        else:
            for metric_config in self.config.metric_list:
                if "metric" not in metric_config:
                    raise ValueError(
                        "'metric' key not provided for an entry in 'metric_list', must be specified!"
                    )
                metric_name = metric_config["metric"]
                kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key
                    not in ["metric", "aggregation", "higher_is_better", "hf_evaluate"]
                }
                hf_evaluate_metric = (
                    "hf_evaluate" in metric_config
                    and metric_config["hf_evaluate"] is True
                )

                if self.config.process_results is not None:
                    self._metric_fn_list[metric_name] = None
                    self._metric_fn_kwargs[metric_name] = {}
                elif callable(metric_name):
                    metric_fn = metric_name.__call__
                    metric_name = metric_name.__name__
                    self._metric_fn_list[metric_name] = metric_fn
                    self._metric_fn_kwargs[metric_name] = kwargs
                else:
                    self._metric_fn_list[metric_name] = get_metric(
                        metric_name, hf_evaluate_metric
                    )
                    self._metric_fn_kwargs[metric_name] = kwargs

                if "aggregation" in metric_config:
                    agg_name = metric_config["aggregation"]
                    if isinstance(agg_name, str):
                        self._aggregation_list[metric_name] = get_aggregation(agg_name)
                    elif callable(agg_name):  # noqa: E721
                        self._aggregation_list[metric_name] = metric_config[
                            "aggregation"
                        ]
                else:
                    INV_AGG_REGISTRY = {v: k for k, v in AGGREGATION_REGISTRY.items()}
                    metric_agg = get_metric_aggregation(metric_name)
                    eval_logger.warning(
                        f"[Task: {self.config.task}] metric {metric_name} is defined, but aggregation is not. "
                        f"using default "
                        f"aggregation={INV_AGG_REGISTRY[metric_agg]}"
                    )
                    self._aggregation_list[metric_name] = metric_agg

                if "higher_is_better" in metric_config:
                    self._higher_is_better[metric_name] = metric_config[
                        "higher_is_better"
                    ]
                else:
                    eval_logger.warning(
                        f"[Task: {self.config.task}] metric {metric_name} is defined, but higher_is_better is not. "
                        f"using default "
                        f"higher_is_better={is_higher_better(metric_name)}"
                    )
                    self._higher_is_better[metric_name] = is_higher_better(metric_name)

            super().__init__(
                data_dir,
                cache_dir,
                config,
            )

    def setup_dataset_paths(self):
        def _setup_image_folder():
            if not self.config.dataset_path:
                raise ValueError(
                    "No dataset_path set; can't construct ImageFolder data"
                )
            self.dataset_paths = {}
            if self.has_training_imgs():
                self.dataset_paths[self.config.training_split] = os.path.join(
                    self.config.dataset_path,
                    self.config.training_split,  # pyright: ignore
                )
            if self.has_validation_imgs():
                self.dataset_paths[self.config.validation_split] = os.path.join(
                    self.config.dataset_path,
                    self.config.validation_split,  # pyright: ignore
                )
            if self.has_test_imgs():
                self.dataset_paths[self.config.test_split] = os.path.join(
                    self.config.dataset_path,
                    self.config.test_split,  # pyright: ignore
                )
            self.dataset = {}

        match self.config.dataset_type:
            case "image-folder":
                _setup_image_folder()
            case _:
                raise NotImplementedError(
                    f"No handler for dataset type {self.config.dataset_type}"
                )

    def setup_dataset(self, transform):
        for k, v in self.dataset_paths.items():
            self.dataset[k] = timm.data.dataset.ImageDataset(v, transform=transform)

    def has_training_imgs(self) -> bool:
        if self.config.training_split is not None:
            return True
        else:
            return False

    def has_validation_imgs(self) -> bool:
        if self.config.validation_split is not None:
            return True
        else:
            return False

    def has_test_imgs(self) -> bool:
        if self.config.test_split is not None:
            return True
        else:
            return False

    def training_imgs(
        self,
    ) -> timm.data.dataset.ImageDataset:
        if self.has_training_imgs():
            return self.dataset[self.config.training_split]

    def validation_imgs(
        self,
    ) -> timm.data.dataset.ImageDataset:
        if self.has_validation_imgs():
            return self.dataset[self.config.validation_split]

    def test_imgs(self) -> timm.data.dataset.ImageDataset:
        if self.has_test_imgs():
            return self.dataset[self.config.test_split]

    def process_results(self, output, label):
        assert callable(self.config.process_results)
        return self.config.process_results(output, label)

    def aggregation(self) -> dict:
        return self._aggregation_list

    def higher_is_better(self) -> dict:
        return self._higher_is_better

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @property
    def task_name(self) -> Any:
        return getattr(self.config, "task", None)

    def __repr__(self):
        return (
            f"ConfigurableTask(task_name={getattr(self.config, 'task', None)}, "
            f"output_type={self.OUTPUT_TYPE})"
        )
