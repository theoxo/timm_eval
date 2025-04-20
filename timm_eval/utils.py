import collections
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from itertools import islice
from typing import Any, Callable, List

import torch
import numpy as np
import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("timm-eval")

SPACING = " " * 47

HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def escaped_split(text, sep_char, maxsplit=-1):
    """Split text into a list on occurrences of the given separation
    character `sep_char`. The separation character may be escaped by a
    backslash to avoid splitting at that location.

    The separation character must be a string of size 1.

    If `maxsplit` is given, at most `maxsplit` splits are done (thus,
    the list will have at most `maxsplit + 1` elements). If `maxsplit`
    is not specified or less than 0, then there is no limit on the
    number of splits (all possible splits are made).
    """
    assert len(sep_char) == 1, (
        "separation string must be a single character for escaped splitting"
    )

    if maxsplit == 0:
        return text
    maxsplit = max(0, maxsplit)

    return re.split(r"(?<!\\)" + sep_char, text, maxsplit)


def parse_list_string(list_str):
    """Parse a string representation of a list into actual Python list"""
    if not (list_str.startswith("[") and list_str.endswith("]")):
        return list_str

    # Remove brackets and split by comma
    items = list_str[1:-1].split(",")
    return [handle_arg_string(item.strip()) for item in items]


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.lower() == "none":
        return None
    elif "float" in arg and hasattr(torch, arg):
        # value is a torch dtype
        return getattr(torch, arg)
    elif arg.startswith("[") and arg.endswith("]"):
        return parse_list_string(arg)
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def sanitize_list(sub):
    """
    Takes possible nested list and recursively converts all inner component to strings
    """
    if isinstance(sub, list):
        return [sanitize_list(item) for item in sub]
    if isinstance(sub, tuple):
        return tuple(sanitize_list(item) for item in sub)
    else:
        return str(sub)


def simple_parse_args_string(args_string):
    """
    Parses argument strings with support for nested dictionaries and lists.
    Examples:
        "args1=val1,arg2=val2" -> {"args1": "val1", "arg2": "val2"}
        "config:key=[1,2,3]" -> {"config": {"key": [1, 2, 3]}}
        "config:k1=v1,config:k2=v2" -> {"config": {"k1": "v1", "k2": "v2"}}
    """
    args_string = args_string.strip()
    if not args_string:
        return {}

    # Split on commas that are not inside square brackets
    arg_list = [arg for arg in re.split(r",(?![^\[]*\])", args_string) if arg]
    result_dict = {}

    for arg in arg_list:
        key, value = arg.split("=")
        # Handle nested dictionary keys
        if ":" in key:
            top_key, nested_key = key.split(":", 1)
            if top_key not in result_dict:
                result_dict[top_key] = {}
            result_dict[top_key][nested_key] = handle_arg_string(value)
        else:
            result_dict[key] = handle_arg_string(value)

    return result_dict


def join_iters(iters):
    for iter in iters:
        yield from iter


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_file_task_name(filename: str) -> str:
    """
    Given the sample results filenames, extracts and returns the task name.
    """
    return filename[filename.find("_") + 1 : filename.rfind("_")]


def get_file_datetime(filename: str) -> str:
    """
    Given the results and sample results filenames, extracts and returns the datetime.
    """
    return filename[filename.rfind("_") + 1 :].replace(".jsonl", "")


def sanitize_model_name(model_name: str) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)


def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)


def get_latest_filename(filenames: List[str]) -> str:
    """
    Given a list of filenames, returns the filename with the latest datetime.
    """
    return max(filenames, key=lambda f: get_file_datetime(f))


def get_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to aggregated results.
    """
    return [f for f in filenames if "/results_" in f and ".json" in f]


def get_sample_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to sample results.
    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class Reorderer:
    def __init__(self, arr: List[Any], fn: Callable) -> None:
        """Reorder an array according to some function

        Args:
            arr (List[Any]): The initial array
            fn (Callable[[Any], Any]): A function to determine the priority of elements
        """
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        # arr = [([y[0] for y in x], x[0][1]) for x in arr]
        # TODO: overhaul reorderer. It currently grouped requests by content but we don't want this
        arr = [([y[0]], x[0][1]) for x in arr for y in x]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        """Gets the reordered array

        Returns:
            List[Any]: The reordered array
        """
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        """Restores the original order of a new array based on the old array's order

        Args:
            newarr (List[Any]): The array to be restored

        Returns:
            List[Any]: The array restored to the original order
        """
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def make_table(result_dict, column: str = "results", sort_results: bool = False):
    """Generate table of results."""
    from pytablewriter import LatexTableWriter, MarkdownTableWriter

    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        column_name,
        "Version",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically by task or group name.
        # NOTE: we default here to false, because order matters for multi-level table printing a la mmlu.
        # sorting here would mess that up
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "    N/A")
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for m, v in metric_items:
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = "%.4f" % v if isinstance(v, float) else v

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                se = "   N/A" if se == "N/A" else "%.4f" % se
                values.append([k, version, m, hib, v, "±", se])
            else:
                values.append([k, version, m, hib, v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "timm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


def ignore_constructor(loader, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        constructor_fn = import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(
    loader=BaseLoader, undefined=StrictUndefined, keep_trailing_newline=True
)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)


class SlicedIterator:
    def __init__(self, raw_iterator, rank, limit, world_size):
        self.it = islice(raw_iterator, rank, limit, world_size)
        self._len = len(raw_iterator) // world_size
        if limit is not None:
            self._len = min(self._len, limit)

    def __iter__(self):
        return self.it

    def __len__(self):
        return self._len


def create_iterator(raw_iterator, *, rank=0, world_size=1, limit=None):
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return SlicedIterator(raw_iterator, rank, limit, world_size)


def weighted_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore


class KMeans(torch.nn.Module):
    """
    Minimal PyTorch k-means implementation.
    """

    def __init__(
        self,
        gen: torch.Generator,
        max_iter: int = 100,
        n_clusters: int = 8,
    ):
        super(KMeans, self).__init__()
        self.max_iter = max_iter
        self.gen = gen
        self.n_clusters = n_clusters

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (*BS, N, D)

        Returns:
            torch.Tensor clusters locations (*BS, n_clusters, D)

        """
        if not self.n_clusters:
            # do full rank clustering, so just return a copy of x really
            return x

        init_centers = self._init_rnd(x)

        cluster_centers = self._cluster(x, init_centers)

        return cluster_centers

    def _init_rnd(self, x: torch.Tensor) -> torch.Tensor:
        """Choose self.n_clusters random nodes as initial centers.

        Args:
            x: (*BS, N, D)

        Returns:
            centers: (*BS, self.n_clusters, D)

        """
        x_shape = x.shape  # *BS, N, D: at least one batch dimension + N (number of points) + D (data dimensionality)
        n = x_shape[-2]
        indices = torch.randint(
            0, n, (self.n_clusters,), generator=self.gen, device=x.device
        )
        centers = x.index_select(dim=-2, index=indices)

        return centers

    def _cluster(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Perform KMeans clustering optimized for torch.compile (so, no convergence check).

        Args:
            x: input features/coordinates (*BS, N, D)
            centers: initial cluster centers (*BS, n_clusters, D)

        Returns:
            torch.Tensor: final cluster centers (*BS, n_clusters, D)
        """

        for _ in range(self.max_iter):
            # Compute distances
            distances = torch.cdist(x, centers)

            # Assign points to nearest center
            assignments = torch.argmin(distances, dim=-1)

            # Convert assignments to one-hot encoding
            assignments_onehot = torch.nn.functional.one_hot(
                assignments, num_classes=self.n_clusters
            ).float()

            # Compute new centers
            cluster_sizes = assignments_onehot.sum(dim=-2, keepdim=True).transpose(
                -1, -2
            )
            centers = torch.einsum("...nd,...nk->...kd", x, assignments_onehot)
            centers /= cluster_sizes.clamp(min=1)

        return centers

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_iter: {self.max_iter}, "
            f"n_clusters: {self.n_clusters}, "
            f"seed: {self.seed})"
        )


def compile_and_warmup(model, input_size, batch_size):
    """
    Compile the Vision Transformer model and perform a warmup run.

    Args:
        model: The Vision Transformer model to compile and warmup.
        input_size (Tuple[int]): The size of the input images.
        batch_size (int): The batch size to use for the synthetic input.

    Returns:
        The compiled model.
    """
    compiled_model = torch.compile(
       model, fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False
    )
    #for block in model.model.blocks:
    #    block.attn = torch.compile(
    #        block.attn,
    #        fullgraph=True,
    #        dynamic=False,
    #    )

    # Perform a warmup run with synthetic data
    with torch.no_grad():
        # Create synthetic input of same shape as real data
        shape = (batch_size, *input_size)
        synthetic_input = torch.randn(
            shape,
            device=next(model.model.parameters()).device,
        )

        eval_logger.debug(
            f"Created synthetic input of shape {synthetic_input.shape} to warm up model"
        )

        # Run the model
        _ = model.model(synthetic_input)

    return compiled_model
