from src.generative_cav.loggers import LOGGER
from src.generative_cav.framework.VisualTCAV import (GlobalVisualTCAV,
                                                     LocalVisualTCAV,
                                                     cosine_similarity)


def get_cosine_similarity(
    local_visual_tcav, concept_name_1, concept_name_2, layer_name
):
    """
    Compute the cosine similarity between two concepts.
    """
    x = local_visual_tcav.getCAVs(layer_name, concept_name_1).direction
    y = local_visual_tcav.getCAVs(layer_name, concept_name_2).direction
    return cosine_similarity(x, y)


def run_local_visual_tcav(test_image_filename, concept_group, model):
    local_visual_tcav = LocalVisualTCAV(
        test_image_filename=test_image_filename,
        n_classes=3,
        m_steps=50,
        batch_size=100,
        model=model.model_object,
    )
    local_visual_tcav.predict()
    local_visual_tcav.setLayers(layer_names=model.layers)
    local_visual_tcav.setConcepts(
        concept_names=[concept_group.true_label] + concept_group.generated
    )
    local_visual_tcav.explain()
    local_visual_tcav.plot()
    ## Cosine Similarities
    similarities = {}
    for layer in model.layers:
        for concept in concept_group.generated:
            similarities[model.name, layer, concept] = get_cosine_similarity(
                local_visual_tcav,
                concept_name_1=concept_group.true_label,
                concept_name_2=concept,
                layer_name=layer,
            )
    print(f"Cosine similarities: {similarities}")
    LOGGER.log_metrics(similarities)


def run_global_visual_tcav(
    test_images_folder, object_class, concept_group, model
):
    global_visual_tcav = GlobalVisualTCAV(
        test_images_folder=test_images_folder,
        target_class=object_class.name,
        m_steps=30,
        batch_size=100,
        model=model.model_object,
    )
    global_visual_tcav.setLayers(layer_names=model.layers)
    global_visual_tcav.setConcepts(
        concept_names=[concept_group.true_label] + concept_group.generated
    )
    global_visual_tcav.explain(cache_cav=True, cache_random=True)
    global_visual_tcav.plot()  ## Plots must be saved
