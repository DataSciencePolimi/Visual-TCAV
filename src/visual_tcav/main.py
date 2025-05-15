import argparse

from config_loader import load_config

from src.generative_cav.framework.run_visual_tcav import (
    run_global_visual_tcav, run_local_visual_tcav)
from src.generative_cav.loggers import LOGGER


def run_all_tcavs(object_class, concept_group, model):
    LOGGER.log_text(f"            Running model: {model.name}")
    LOGGER.log_text(f"            on layers: {model.layers}")
    LOGGER.log_text(f"            with preprocessing function: {model.preprocessing_function}")

    run_local_visual_tcav(object_class.example_image, concept_group, model)
    run_global_visual_tcav(object_class.name, object_class, concept_group, model)

    if concept_group.concept_imputation.example:
        run_local_visual_tcav(
            concept_group.concept_imputation.example, concept_group, model
        )
    if concept_group.concept_imputation.folder:
        run_global_visual_tcav(
            concept_group.concept_imputation.folder,
            object_class,
            concept_group,
            model,
        )


def main(config_path: str = "../../config.yaml"):
    config = load_config(config_path)
    LOGGER.log_config(config.model_dump())
    for object_class in config.classes:
        LOGGER.log_text(f"Using class: {object_class.name}")
        for concept_group in object_class.concepts_groups:
            LOGGER.log_text(f"    Selected concept group: {concept_group.true_label}")
            LOGGER.log_text(f"        Analysis will run on generated concepts: {concept_group.generated}")
            for model in config.models:
                run_all_tcavs(object_class, concept_group, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual TCAV analysis using a config file.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config_path)

