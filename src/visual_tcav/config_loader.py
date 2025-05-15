import importlib
from typing import Callable, List, Optional

import yaml
from pydantic import BaseModel, field_validator, Field

from src.visual_tcav.framework.VisualTCAV import Model


class ImputationTask(BaseModel):
    example: Optional[str] = None
    folder: Optional[str] = None


class ConceptGroup(BaseModel):
    true_label: str
    generated: List[str]
    concept_imputation: ImputationTask = Field(default_factory=ImputationTask)

class ClassConfig(BaseModel):
    name: str
    example_image: str
    concepts_groups: List[ConceptGroup]


class ModelConfig(BaseModel):
    name: str
    graph_path_filename: str
    label_path_filename: str
    preprocessing_function: Callable
    max_examples: int
    layers: List[str]

    @field_validator("preprocessing_function", mode="before")
    def validate_preprocessing_function(cls, dotted_path):
        try:
            module_path, function = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, function)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Invalid preprocessing function path: {dotted_path}"
            ) from e

    @property
    def model_object(self):
        return Model(
            model_name=self.name,
            graph_path_filename=self.graph_path_filename,
            label_path_filename=self.label_path_filename,
            preprocessing_function=self.preprocessing_function,
            max_examples=self.max_examples,
        )


class RootConfig(BaseModel):
    classes: List[ClassConfig]
    models: List[ModelConfig]


def load_config(config_path: str = "config.yaml") -> RootConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return RootConfig(**data)
