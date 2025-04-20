from dataclasses import dataclass, field
from typing import Literal


ALL_OUTPUT_TYPES = [
    "logits",
]
OutputType = Literal["logits",]


@dataclass
class Instance:
    request_type: OutputType
    img: dict
    img_id: int
    task_name: str
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
