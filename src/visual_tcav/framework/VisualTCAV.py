#####
#	VisualTCAV
#
#	All rights reserved.
#
#	Main classes
#####


#####
# Imports
#####

# Do not generate "__pycache__" folder
import sys
from pathlib import Path
from src.loggers import LOGGER

sys.dont_write_bytecode = True

import os
from multiprocessing import dummy as multiprocessing

import numpy as np
import PIL.Image
import PIL.ImageFilter
from joblib import dump, load
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from prettytable import PrettyTable
from tqdm import tqdm
import platform

# Tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_probability as tfp
from keras.src.losses import MeanSquaredError

# Keras preprocessing functions
preprocess_resnet_v2 = tf.keras.applications.inception_resnet_v2.preprocess_input
preprocess_v3 = tf.keras.applications.inception_v3.preprocess_input
preprocess_vgg16 = tf.keras.applications.vgg16.preprocess_input


# Utils
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def nth_highest_index(arr, n):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return sorted_arr[n - 1][0]


def contraharmonic_mean(arr, axis=(0, 1)):
    numerator = tf.reduce_sum(tf.square(arr), axis=axis)
    denominator = tf.reduce_sum(arr, axis=axis)
    return tf.divide(numerator, (tf.add(denominator, tf.keras.backend.epsilon())))


#####
# VisualTCAV class
#####


class VisualTCAV:
    ##### Init #####
    def __init__(
        self,
        model,
        visual_tcav_dir="VisualTCAV",
        clear_cache=False,
        batch_size=250,
        models_dir=None,
        cache_dir=None,
        test_images_dir=None,
        concept_images_dir=None,
        random_images_folder=None,
    ):

        # Folders and directories
        self.models_dir = (
            os.path.join(visual_tcav_dir, "models") if not models_dir else models_dir
        )
        self.cache_base_dir = (
            os.path.join(visual_tcav_dir, "cache") if not cache_dir else cache_dir
        )
        self.cache_dir = self.cache_base_dir
        self.test_images_dir = (
            os.path.join(visual_tcav_dir, "test_images")
            if not test_images_dir
            else test_images_dir
        )
        self.concept_images_dir = (
            os.path.join(visual_tcav_dir, "concept_images")
            if not concept_images_dir
            else concept_images_dir
        )
        self.random_images_folder = (
            "random" if not random_images_folder else random_images_folder
        )

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_base_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        os.makedirs(self.concept_images_dir, exist_ok=True)

        self.batch_size = batch_size

        # Model
        self.model = None
        if model:
            self._bindModel(model)

        if clear_cache:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

        # Concepts/Layers attributes
        self.concepts = []
        self.layers = []

        # Computations
        self.computations = {}
        self.random_acts = {}

    # Set a list of concepts
    def setConcepts(self, concept_names):
        self.concepts = []
        for concept_name in concept_names:
            if concept_name not in self.concepts:
                self.concepts.append(concept_name)

    # Set a list of layers
    def setLayers(self, layer_names):
        self.layers = []
        for layer_name in layer_names:
            if layer_name not in self.layers:
                self.layers.append(layer_name)

    ##### Predict #####
    def predict(self, no_sort=False):

        # Checks
        if not isinstance(self, LocalVisualTCAV):
            raise Exception("Please use a local explainer")
        if not self.model:
            raise Exception("Please instantiate a Model first")

        # Predict with the provided model wrapper
        self.predictions = self.model.model_wrapper.get_predictions(
            self.model.preprocessing_function(self.resized_imgs)
        )

        # Sort & add class names
        self.predictions = np.array(
            [
                self._sortTargetClasses(
                    prediction, self.model.model_wrapper.id_to_label, no_sort
                )
                for prediction in self.predictions
            ]
        )

        # Return the classes
        return Predictions(
            self.predictions, self.test_image_filename, self.model.model_name
        )

    #####
    # Private methods
    #####

    # Bind a model
    def _bindModel(self, model):

        # Folders and directories
        model.graph_path_dir = os.path.join(
            self.models_dir, model.model_name, model.graph_path_filename
        )
        model.label_path_dir = os.path.join(
            self.models_dir, model.model_name, model.label_path_filename
        )

        # Wrapper function
        model.model_wrapper = model.model_wrapper(
            model.graph_path_dir, model.label_path_dir, self.batch_size
        )

        # Activate the model
        model.activation_generator = model.activation_generator(
            model_wrapper=model.model_wrapper,
            concept_images_dir=self.concept_images_dir,
            cache_dir=self.cache_dir,
            preprocessing_function=model.preprocessing_function,
            max_examples=model.max_examples,
        )

        # Model's cache dir
        self.cache_dir = os.path.join(self.cache_base_dir, model.model_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Store the model
        self.model = model

    # Reshape a list of predictions
    def _sortTargetClasses(self, predictions, id_to_label, no_sort=False):

        # Reshape
        indexed_arr = list(enumerate(predictions))
        sorted_arr = (
            indexed_arr
            if no_sort
            else sorted(indexed_arr, key=lambda x: x[1], reverse=True)
        )
        return [
            Prediction(
                class_index=sorted_element[0],
                class_name=id_to_label(sorted_element[0]),
                confidence=sorted_element[1],
            )
            for i, sorted_element in enumerate(sorted_arr)
            if i < 10
        ]

    # Utils to compute the integrated gradients
    def _compute_integrated_gradients(self, feature_maps, layer_name, class_index):
        # Alphas and baseline image for interpolating images
        alphas = tf.linspace(
            start=0.0, stop=1.0, num=self.m_steps + 1
        )  # Generate m_steps intervals for riemann approximation
        baseline = tf.zeros(shape=feature_maps.shape)
        # Interpolate images
        interpolated_images = self._interpolate_images(feature_maps, baseline, alphas)
        # VisualTCAV.tf_session.run(
        # Generating gradients
        # if self.model.model_name == "InceptionV3":
        #	grads = np.array([])
        #	for image in interpolated_images:
        #		grads = np.append(grads,
        #			# Grad points in the direction which INCREASES probability of class
        #			self.model.model_wrapper.get_gradient_of_score(np.expand_dims(image, axis=0), layer_name, class_index)[0],
        #		)
        # else:
        grads = self.model.model_wrapper.get_gradient_of_score(
            interpolated_images, layer_name, class_index
        )
        # Compute the gradients
        return tf.math.reduce_mean(
            (np.array(grads)[:-1] + np.array(grads)[1:]) / tf.constant(2.0),
            axis=0,
        )

    # Utils function to interpolate the fmaps
    def _interpolate_images(self, feature_maps, baseline, alphas):
        # Interpolating fmaps
        image = tf.image.convert_image_dtype(feature_maps, tf.float32)
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = tf.subtract(input_x, baseline_x)
        images = tf.add(baseline_x, tf.multiply(alphas_x, delta))
        return images

    # Function to compute the negative examples activations for a given layer
    def _compute_random_activations(self, cache, layer_name):
        # Random activations
        cache_random_acts_path = os.path.join(
            self.cache_dir,
            "rnd_acts_"
            + str(self.model.max_examples)
            + "_"
            + self.random_images_folder
            + "_"
            + layer_name
            + ".joblib",
        )
        if cache and os.path.isfile(cache_random_acts_path):
            random_acts = load(cache_random_acts_path)
        else:
            random_acts = self._compute_random(layer_name)

            # If cache is requested
            if cache:
                dump(random_acts, cache_random_acts_path, compress=3)
        # Return
        return random_acts

    # Compute pooled random
    def _compute_random(self, layer_name):
        feature_maps_for_concept = (
            self.model.activation_generator.get_feature_maps_for_concept(
                self.random_images_folder,
                layer_name,
            )
        )
        return feature_maps_for_concept

    # Function to compute the CAV given a concept & a layer
    def _compute_cavs(self, cache, concept_name, layer_name, random_acts):

        # If cached file exists
        cache_path = os.path.join(
            self.cache_dir,
            "cav_"
            + concept_name
            + "_"
            + str(self.model.max_examples)
            + "_"
            + self.random_images_folder
            + "_"
            + layer_name
            + ".joblib",
        )
        if cache and os.path.isfile(cache_path):
            concept_layer = load(cache_path)
        else:
            # Activations (concept/layer)
            concept_acts = self.model.activation_generator.get_feature_maps_for_concept(
                concept_name,
                layer_name,
            )

            pooled_concept = tf.reduce_mean(concept_acts, axis=(1, 2))

            pooled_random = tf.reduce_mean(random_acts, axis=(1, 2))

            # CAV
            concept_layer = ConceptLayer()

            concept_layer.cav.centroid0 = tf.reduce_mean(pooled_concept, axis=0)
            concept_layer.cav.centroid1 = tf.reduce_mean(pooled_random, axis=0)
            concept_layer.cav.direction = tf.subtract(
                concept_layer.cav.centroid0, concept_layer.cav.centroid1
            )

            emblems = contraharmonic_mean(
                tf.nn.relu(
                    tf.reduce_sum(
                        tf.multiply(
                            concept_layer.cav.direction[None, None, None, :],
                            concept_acts,
                        ),
                        axis=3,
                    )
                ),
                axis=(1, 2),
            )

            negative_emblems = contraharmonic_mean(
                tf.nn.relu(
                    tf.reduce_sum(
                        tf.multiply(
                            concept_layer.cav.direction[None, None, None, :],
                            random_acts,
                        ),
                        axis=3,
                    )
                ),
                axis=(1, 2),
            )

            concept_layer.cav.concept_emblem = tf.cast(
                (
                    tfp.stats.percentile(emblems, 50.0),
                    tfp.stats.percentile(negative_emblems, 50.0),
                ),
                tf.float32,
            )
            # If cache is requested
            if cache:
                dump(concept_layer, cache_path, compress=3)

        # Return
        return concept_layer


#####
# LocalVisualTCAV
#####


class LocalVisualTCAV(VisualTCAV):

    ##### Init #####
    def __init__(
        self,
        test_image_filename,
        m_steps=50,
        n_classes=3,
        target_class=None,
        *args,
        **kwargs,
    ):
        # Super
        super().__init__(**kwargs)

        # Local attributes
        self.test_image_filename = test_image_filename
        self.m_steps = m_steps
        self.target_class = target_class
        if self.target_class is not None:
            self.target_class_index = self.model.model_wrapper.label_to_id(
                self.target_class
            )
        elif not self.model.binary_classification:
            self.n_classes = max(
                np.min([n_classes, len(self.model.model_wrapper.labels), 3]), 1
            )  # Not implemented more than 3
        else:
            self.n_classes = 2  # add check that it's actually binary
        self.test_images_dir = os.path.join(
            self.test_images_dir, self.test_image_filename
        )
        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

        self.predictions = []
        self.computations = {}

        # Load and resize the image/images
        self.imgs = np.array(
            [
                PIL.Image.open(tf.io.gfile.GFile(self.test_images_dir, "rb")).convert(
                    "RGB"
                )
            ]
        )
        self.resized_imgs = np.array(
            [
                PIL.Image.open(tf.io.gfile.GFile(self.test_images_dir, "rb"))
                .convert("RGB")
                .resize(self.resized_imgs_size, PIL.Image.BILINEAR)
            ]
        )

    ##### Explain #####
    def explain(self, cache_cav=True, cache_random=True, cav_only=False):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")

        # Reset the computation variable
        self.computations = {}

        # For each layer
        for layer_name in tqdm(self.layers, desc="Layers", position=0):
            self.computations[layer_name] = {}

            # Random activations
            random_acts = self._compute_random_activations(cache_random, layer_name)

            # Compute the feature maps
            feature_maps = self.model.model_wrapper.get_feature_maps(
                self.model.preprocessing_function(self.resized_imgs), layer_name
            )[0]

            # Compute the CAVs
            # Note:	computing the direction of the cav using the GAP of the concept's activations and the GAP of the
            # 		random's activations is equivalent to computing the GAP of the direction of the cav, obtained with the
            # 		concept's activations and the random's activations
            # 		cav.direction = GAP(cav.centroid0 - cav.centroid1) = GAP(cav.centroid0) - GAP(cav.centroid1)
            for concept_name in self.concepts:

                # CAVs
                concept_layer = self._compute_cavs(
                    cache_cav, concept_name, layer_name, random_acts
                )

                if not cav_only:

                    # Concept map
                    concept_layer.concept_map = tf.nn.relu(
                        tf.math.reduce_sum(
                            tf.multiply(
                                concept_layer.cav.direction[None, None, :], feature_maps
                            ),
                            axis=2,
                        )
                    )

                    # Normalize Concept Map
                    if (
                        concept_layer.cav.concept_emblem[0]
                        > concept_layer.cav.concept_emblem[1]
                    ):
                        concept_layer.concept_map = tf.where(
                            concept_layer.concept_map
                            > concept_layer.cav.concept_emblem[0],
                            concept_layer.cav.concept_emblem[0],
                            concept_layer.concept_map,
                        )
                        concept_layer.concept_map = tf.where(
                            concept_layer.concept_map
                            < concept_layer.cav.concept_emblem[1],
                            concept_layer.cav.concept_emblem[1],
                            concept_layer.concept_map,
                        )
                        concept_layer.concept_map = (
                            concept_layer.concept_map
                            - concept_layer.cav.concept_emblem[1]
                        ) / (
                            concept_layer.cav.concept_emblem[0]
                            - concept_layer.cav.concept_emblem[1]
                        )
                    else:
                        concept_layer.concept_map = tf.multiply(
                            concept_layer.concept_map, 0
                        )

                # Save the partial computations
                self.computations[layer_name][concept_name] = concept_layer

            if not cav_only:
                # Compute integrated gradients and attributions
                attributions = {}
                for n_class in range(self.n_classes):
                    if not self.model.binary_classification:
                        logits = self.model.model_wrapper.get_logits(
                            np.expand_dims(feature_maps, axis=0), layer_name
                        )[0]
                        logits_baseline = self.model.model_wrapper.get_logits(
                            np.expand_dims(tf.zeros(shape=feature_maps.shape), axis=0),
                            layer_name,
                        )[0]

                        ig_expected = tf.nn.relu(tf.subtract(logits, logits_baseline))

                        ig_expected_max_value = tf.reduce_max(ig_expected)
                        if ig_expected_max_value > 0:
                            ig_expected_norm = tf.divide(
                                ig_expected, ig_expected_max_value
                            )
                        else:
                            ig_expected_norm = ig_expected

                        if self.target_class is not None:
                            ig_expected_class = ig_expected_norm[
                                self.target_class_index
                            ]
                        # elif self.model.binary_classification:
                        # 	ig_expected_class = ig_expected_norm[self.predictions[0][0].class_index]
                        else:
                            ig_expected_class = ig_expected_norm[
                                self.predictions[0][n_class].class_index
                            ]

                    # Compute attributions
                    if self.target_class is not None:
                        ig = self._compute_integrated_gradients(
                            feature_maps, layer_name, self.target_class_index
                        )
                    elif self.model.binary_classification:
                        ig = self._compute_integrated_gradients(
                            feature_maps, layer_name, self.predictions[0][0].class_index
                        )
                    else:
                        ig = self._compute_integrated_gradients(
                            feature_maps,
                            layer_name,
                            self.predictions[0][n_class].class_index,
                        )

                    if self.model.binary_classification:
                        binary_attributions = tf.multiply(ig, feature_maps)
                        virtual_logit_0 = tf.reduce_sum(tf.nn.relu(binary_attributions))
                        virtual_logit_1 = tf.reduce_sum(
                            tf.nn.relu(-binary_attributions)
                        )
                        max_virtual_logit = max(virtual_logit_0, virtual_logit_1)
                        if max_virtual_logit > 0:
                            virtual_logit_0 /= max_virtual_logit
                            virtual_logit_1 /= max_virtual_logit
                        if n_class == 0:
                            attributions[n_class] = tf.nn.relu(binary_attributions)
                            attributions[n_class] = tf.multiply(
                                tf.divide(
                                    attributions[n_class],
                                    tf.add(
                                        tf.reduce_sum(attributions[n_class]),
                                        tf.keras.backend.epsilon(),
                                    ),
                                ),
                                virtual_logit_0,
                            )
                        else:
                            attributions[n_class] = tf.nn.relu(-binary_attributions)
                            attributions[n_class] = tf.multiply(
                                tf.divide(
                                    attributions[n_class],
                                    tf.add(
                                        tf.reduce_sum(attributions[n_class]),
                                        tf.keras.backend.epsilon(),
                                    ),
                                ),
                                virtual_logit_1,
                            )

                    else:
                        attributions[n_class] = tf.nn.relu(
                            tf.multiply(ig, feature_maps)
                        )
                        attributions[n_class] = tf.multiply(
                            tf.divide(
                                attributions[n_class],
                                tf.add(
                                    tf.reduce_sum(attributions[n_class]),
                                    tf.keras.backend.epsilon(),
                                ),
                            ),
                            ig_expected_class,
                        )

                # Iterate again on concepts and n_classes
                for concept_name in self.concepts:
                    for n_class in range(self.n_classes):
                        # Mask attributions
                        masked_attributions = tf.multiply(
                            attributions[n_class],
                            self.computations[layer_name][concept_name].concept_map[
                                :, :, None
                            ],
                        )
                        pooled_masked_attributions = tf.reduce_sum(
                            masked_attributions, axis=(0, 1)
                        )

                        # Pooled & normalized CAV
                        if tf.reduce_min(feature_maps) < 0:
                            pooled_cav_norm = tf.nn.relu(
                                tf.multiply(
                                    self.computations[layer_name][
                                        concept_name
                                    ].cav.direction,
                                    tf.where(
                                        tf.reduce_sum(
                                            tf.multiply(
                                                feature_maps,
                                                self.computations[layer_name][
                                                    concept_name
                                                ].concept_map[:, :, None],
                                            ),
                                            axis=(0, 1),
                                        )
                                        < 0,
                                        -1.0,
                                        1.0,
                                    ),
                                )
                            )
                        else:
                            pooled_cav_norm = tf.nn.relu(
                                self.computations[layer_name][
                                    concept_name
                                ].cav.direction
                            )

                        max_cav = tf.reduce_max(pooled_cav_norm)
                        if max_cav > 0:
                            pooled_cav_norm = tf.divide(
                                pooled_cav_norm, tf.reduce_max(pooled_cav_norm)
                            )

                        # Compute and save concept attributions
                        self.computations[layer_name][concept_name].attributions[
                            n_class
                        ] = tf.tensordot(
                            pooled_cav_norm, pooled_masked_attributions, axes=1
                        )

    ##### Plot heatmaps and information #####
    def plot(self, paper=False):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")
        if not self.computations:
            raise Exception("Please let the model explain first")

        # Escaping
        model_name_esc = self.model.model_name.replace("_", r"\_")

        # Iterate over the concepts
        for concept_name in self.concepts:

            # Escaping
            concept_name_esc = concept_name.replace("_", r"\_")

            if not paper:
                fig = plt.figure(
                    figsize=(
                        4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1),
                        7,
                    )
                )
                gs = GridSpec(3, len(self.layers) * 3, height_ratios=[2, 5, 2])
                fig.suptitle(
                    rf"$\mathbfit{{{model_name_esc}}}$ architecture, $\mathbfit{{{concept_name_esc}}}$ concept",
                    fontsize=10 + 1.5 * len(self.layers),
                )
            else:
                fig = plt.figure(
                    figsize=(
                        4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1),
                        6,
                    )
                )  # + (0 if self.imgs[0].shape[0] < self.imgs[0].shape[1] else 1)))
                gs = GridSpec(2, len(self.layers) * 3, height_ratios=[1, 3.5])

            # Examples of concepts
            if not paper:
                concept_images = self.model.activation_generator.get_images_for_concept(
                    concept_name, False
                )
                for i in range(min(len(concept_images), len(self.layers) * 3)):
                    fig.add_subplot(gs[2, i])
                    plt.imshow(concept_images[i])
                    plt.tight_layout()
                    plt.axis("off")

            # Iterate over the layers
            for j, layer_name in enumerate(self.layers):

                # Escaping
                layer_description = "" if len(layer_name) > 11 else "layer"
                layer_name_esc = layer_name.replace("_", r"\_")

                # Obtain concept
                concept_layer = self.computations[layer_name][concept_name]

                # Obtain heatmap
                max_value = np.max(concept_layer.concept_map)
                heatmap = tf.image.resize(
                    np.expand_dims(concept_layer.concept_map, axis=2),
                    [self.imgs[0].shape[0], self.imgs[0].shape[1]],
                )
                heatmap = np.reshape(heatmap, (heatmap.shape[0], heatmap.shape[1]))

                # Subplot
                fig.add_subplot(gs[1, j * 3 : (j + 1) * 3])
                plt.imshow(self.imgs[0])

                # Blurring
                heatmap = (
                    np.array(
                        PIL.Image.fromarray(np.uint8(heatmap * 255), "L").filter(
                            PIL.ImageFilter.GaussianBlur(radius=20)
                        )
                    )
                    / 255
                )

                if np.max(heatmap) > 0 and np.max(heatmap) < max_value:
                    heatmap = (heatmap / np.max(heatmap)) * max_value

                colormap.imshow(heatmap)
                if not paper:
                    plt.title(f"", fontsize=1)
                plt.tight_layout()
                plt.axis("off")

                # Subplot
                fig.add_subplot(gs[0, j * 3 : (j + 1) * 3])
                plt.title(
                    rf"$\mathbfit{{{layer_name_esc}}}$ {layer_description}",
                    fontsize=9 + 1.5 * len(self.layers),
                    y=0.95,
                )
                plt.tight_layout()
                rows = []
                for c in range(self.n_classes):
                    attribution = concept_layer.attributions[c]
                    if self.target_class is not None:
                        class_name = self.target_class.replace("-", " ")
                    elif self.model.binary_classification and c == 1:
                        class_name = "Not " + self.predictions[0][0].class_name.replace(
                            "-", " "
                        )
                    else:
                        class_name = self.predictions[0][c].class_name.replace("-", " ")
                    if not paper or True:  # temp
                        if len(class_name) > 12:
                            class_name = class_name[:12] + "‥"
                    class_name = class_name.replace("_", r"\_").replace(" ", r"\ ")
                    row = []
                    row.append(rf"$\mathit{{{class_name}}}$")
                    if paper:
                        attribution = (
                            f"{attribution:.2g}"
                            if (attribution >= 0.001 or attribution == 0.0)
                            else f"{attribution:.1e}"
                        )
                    else:
                        attribution = (
                            f"{attribution:.2g}"
                            if attribution >= 0.001
                            else f"{attribution:.1e}"
                        )
                    attribution = attribution.replace("e-0", "e-").replace("-", "{-}")
                    row.append(rf"$\mathbf{{{attribution}}}$")
                    rows.append(row)

                cols = [rf"$\mathbf{{Class}}$", rf"$\mathbf{{Attrib.}}$"]
                table = plt.table(
                    cellText=rows,
                    rowLabels=[f"" for c in range(self.n_classes)],
                    colLabels=cols,
                    rowColours=["silver"] * 10,
                    colColours=["silver"] * 10,
                    cellLoc="center",
                    rowLoc="center",
                    loc="center",
                    edges="BRTL",
                )
                cellDict = table.get_celld()
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 0)].set_width(0.625)
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 1)].set_width(0.375)
                for i in range(0, len(cols)):
                    cellDict[(0, i)].set_height(0.2)
                    for j in range(1, self.n_classes + 1):
                        cellDict[(j, i)].set_height(0.2)

                # Set font size
                if paper:
                    table.auto_set_font_size(False)
                    table.set_fontsize(4.5 + 1.75 * len(self.layers))
                else:
                    table.set_fontsize(9 + 1.75 * len(self.layers))

                plt.tight_layout()
                plt.axis("off")

            # Show
            fig.tight_layout()
            LOGGER.log_image(plt, self.concepts[0])
            plt.show()

    ##### Get CAVs #####
    def getCAVs(self, layer_name, concept_name):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")
        if not self.computations:
            raise Exception("Please let the model explain first")

        return self.computations[layer_name][concept_name].cav


#####
# GlobalVisualTCAV
#####


class GlobalVisualTCAV(VisualTCAV):

    ##### Init #####
    def __init__(
        self,
        target_class,
        test_images_folder,
        m_steps=50,
        compute_negative_class=False,
        *args,
        **kwargs,
    ):

        # Super
        super().__init__(**kwargs)

        # Local attributes
        self.m_steps = m_steps
        self.target_class = target_class
        self.compute_negative_class = compute_negative_class
        self.test_images_folder = test_images_folder
        self.test_image_filename = test_images_folder
        self.class_index = self.model.model_wrapper.label_to_id(target_class)

        # self.test_images_dir = os.path.join(self.test_images_dir, self.test_images_folder)
        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

        self.predictions = []
        self.stats = {}

    ##### Explain #####
    def explain(self, cache_cav=True, cache_random=True):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")

        # Reset the computation variable
        self.stats = {}

        # For each layer
        for layer_name in tqdm(self.layers, desc="Layers", position=0):
            self.stats[layer_name] = {}

            # Random activations
            random_acts = self._compute_random_activations(cache_random, layer_name)

            # Compute the feature_maps for each class
            class_feature_maps = self.computeFeatureMaps(layer_name)

            # For each concept
            cavs = {}
            attribution_list = {}
            for concept_name in self.concepts:
                # CAVs
                concept_layer = self._compute_cavs(
                    cache_cav, concept_name, layer_name, random_acts
                )

                # Save the partial computations
                cavs[concept_name] = concept_layer
                attribution_list[concept_name] = {}

            # For each image
            for cl, feature_maps in enumerate(
                tqdm(class_feature_maps, desc="Attributions", position=1)
            ):
                """
                if self.target_class is not None:
                    ig_expected_class = ig_expected_norm[self.target_class_index]
                elif self.model.binary_classification:
                    ig_expected_class = ig_expected_norm[self.predictions[0][0].class_index]
                else:
                    ig_expected_class = ig_expected_norm[self.predictions[0][n_class].class_index]

                # Compute attributions
                if self.target_class is not None:
                    ig = self._compute_integrated_gradients(feature_maps, layer_name, self.target_class_index)
                elif self.model.binary_classification:
                    ig = self._compute_integrated_gradients(feature_maps, layer_name, self.predictions[0][0].class_index)
                else:
                    ig = self._compute_integrated_gradients(feature_maps, layer_name, self.predictions[0][n_class].class_index)

                if self.model.binary_classification and n_class == 1:
                    attributions[n_class] = tf.nn.relu(-tf.multiply(ig, feature_maps))
                else:
                    attributions[n_class] = tf.nn.relu(tf.multiply(ig, feature_maps))
                """
                if not self.model.binary_classification:
                    # Compute logits
                    logits = self.model.model_wrapper.get_logits(
                        np.expand_dims(feature_maps, axis=0), layer_name
                    )[0]
                    logits_baseline = self.model.model_wrapper.get_logits(
                        np.expand_dims(tf.zeros(shape=feature_maps.shape), axis=0),
                        layer_name,
                    )[0]

                    ig_expected = tf.nn.relu(tf.subtract(logits, logits_baseline))

                    ig_expected_max_value = tf.reduce_max(ig_expected)
                    if ig_expected_max_value > 0:
                        ig_expected_norm = tf.divide(ig_expected, ig_expected_max_value)
                    else:
                        ig_expected_norm = ig_expected

                    ig_expected_class = ig_expected_norm[self.class_index]

                # Compute attributions
                ig = self._compute_integrated_gradients(
                    feature_maps, layer_name, self.class_index
                )
                if (
                    self.model.binary_classification
                ):  # and self.compute_negative_class == True:
                    # attributions = tf.nn.relu(-tf.multiply(ig, feature_maps))
                    binary_attributions = tf.multiply(ig, feature_maps)
                    virtual_logit_0 = tf.reduce_sum(tf.nn.relu(binary_attributions))
                    virtual_logit_1 = tf.reduce_sum(tf.nn.relu(-binary_attributions))
                    max_virtual_logit = max(virtual_logit_0, virtual_logit_1)
                    if max_virtual_logit > 0:
                        virtual_logit_0 /= max_virtual_logit
                        virtual_logit_1 /= max_virtual_logit
                    if not self.compute_negative_class:
                        attributions = tf.nn.relu(binary_attributions)
                        attributions = tf.multiply(
                            tf.divide(
                                attributions,
                                tf.add(
                                    tf.reduce_sum(attributions),
                                    tf.keras.backend.epsilon(),
                                ),
                            ),
                            virtual_logit_0,
                        )
                    else:
                        attributions = tf.nn.relu(-binary_attributions)
                        attributions = tf.multiply(
                            tf.divide(
                                attributions,
                                tf.add(
                                    tf.reduce_sum(attributions),
                                    tf.keras.backend.epsilon(),
                                ),
                            ),
                            virtual_logit_1,
                        )
                else:
                    attributions = tf.nn.relu(tf.multiply(ig, feature_maps))
                    attributions = tf.multiply(
                        tf.divide(
                            attributions,
                            tf.add(
                                tf.reduce_sum(attributions), tf.keras.backend.epsilon()
                            ),
                        ),
                        ig_expected_class,
                    )

                # Again for each concept
                for concept_name in self.concepts:

                    # Concept map
                    concept_map = tf.nn.relu(
                        tf.math.reduce_sum(
                            tf.multiply(
                                cavs[concept_name].cav.direction[None, None, :],
                                feature_maps,
                            ),
                            axis=2,
                        )
                    )

                    # Normalize Concept Map
                    if (
                        cavs[concept_name].cav.concept_emblem[0]
                        > cavs[concept_name].cav.concept_emblem[1]
                    ):
                        concept_map = tf.where(
                            concept_map > cavs[concept_name].cav.concept_emblem[0],
                            cavs[concept_name].cav.concept_emblem[0],
                            concept_map,
                        )
                        concept_map = tf.where(
                            concept_map < cavs[concept_name].cav.concept_emblem[1],
                            cavs[concept_name].cav.concept_emblem[1],
                            concept_map,
                        )
                        concept_map = (
                            concept_map - cavs[concept_name].cav.concept_emblem[1]
                        ) / (
                            cavs[concept_name].cav.concept_emblem[0]
                            - cavs[concept_name].cav.concept_emblem[1]
                        )
                    else:
                        concept_map = tf.multiply(concept_map, 0)

                    # Mask attributions
                    pooled_masked_attributions = tf.reduce_sum(
                        tf.multiply(attributions, concept_map[:, :, None]), axis=(0, 1)
                    )

                    # Pooled & normalized CAV
                    if tf.reduce_min(feature_maps) < 0:
                        pooled_cav_norm = tf.nn.relu(
                            tf.multiply(
                                cavs[concept_name].cav.direction,
                                tf.where(
                                    tf.reduce_sum(
                                        tf.multiply(
                                            feature_maps, concept_map[:, :, None]
                                        ),
                                        axis=(0, 1),
                                    )
                                    < 0,
                                    -1.0,
                                    1.0,
                                ),
                            )
                        )
                    else:
                        pooled_cav_norm = tf.nn.relu(cavs[concept_name].cav.direction)

                    max_cav = tf.reduce_max(pooled_cav_norm)
                    if max_cav > 0:
                        pooled_cav_norm = tf.divide(
                            pooled_cav_norm, tf.reduce_max(pooled_cav_norm)
                        )

                    # Compute and save concept attributions
                    attribution_list[concept_name][cl] = tf.tensordot(
                        pooled_cav_norm, pooled_masked_attributions, axes=1
                    )

            # Again for each concept
            for concept_name in self.concepts:
                # Compute stats
                self.stats[layer_name][concept_name] = Stat(
                    list(attribution_list[concept_name].values())
                )

            # Clear memory
            del cavs
            del attribution_list

    ##### Plot graphs and information #####
    def plot(self, colormap="viridis", paper=False, save_to=None):
        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not self.stats:
            raise Exception("Please let the model explain first")

        # Colors
        cmap = plt.get_cmap(colormap)

        # Escaping
        model_name_esc = (
            self.model.model_name.replace("_", r"\_")
            .replace("-", "{-}")
            .replace(" ", "\\text{ }")
        )
        target_class_esc = (
            self.target_class.replace("_", r"\_")
            .replace("-", "{-}")
            .replace(" ", "\\text{ }")
        )
        if self.model.binary_classification and self.compute_negative_class == True:
            target_class_esc = "Not " + target_class_esc
        concept_names = [
            concept.replace("_", r"\_").replace("-", "{-}").replace(" ", r"\\text{ }")
            for concept in self.concepts
        ]

        # Figure
        fig = plt.figure(figsize=(5 + 1 * (len(self.concepts) - 1), 4))
        gs = GridSpec(1, 1, height_ratios=[1])
        fig.suptitle(
            rf"$\mathbfit{{{model_name_esc}}}$ architecture, $\mathbfit{{{target_class_esc}}}$ target class",
            fontsize=12,
        )
        # Subplot
        fig.add_subplot(gs[0])

        # Axes
        x = np.arange(len(self.concepts)) - 0.5

        # Iterate over the concepts
        for i, layer_name in enumerate(self.layers):
            # Indexing
            # color = i / (len(self.layers)-1) if len(self.layers) > 1 else 0.5
            color = (
                cmap(i / (len(self.layers) - 1)) if len(self.layers) > 1 else cmap(0.5)
            )
            width = 0.1
            pos_x = 0.5 + (i - len(self.layers) / 2) * width + width / 2

            # Escaping
            layer_name_esc = (
                layer_name.replace("_", r"\_")
                .replace("-", "{-}")
                .replace(" ", "\\text{ }")
            )

            # Bar
            plt.bar(
                x + pos_x,
                [
                    (
                        self.stats[layer_name][concept_name].begin
                        + self.stats[layer_name][concept_name].end
                    )
                    / 2
                    for concept_name in self.concepts
                ],
                yerr=[
                    max(
                        0,
                        (
                            self.stats[layer_name][concept_name].end
                            - self.stats[layer_name][concept_name].begin
                        )
                        / 2,
                    )
                    for concept_name in self.concepts
                ],
                width=width,
                label=rf"$\mathit{{{layer_name_esc}}}$",
                zorder=2,
                capsize=3.5,
                # color=cmap(((color)/8)*6 + 1/8),
                color=color,
            )

        # Show
        # plt.xlabel('Concept')
        plt.ylabel("Attribution (2σ error)")
        plt.xticks(
            np.arange(len(self.concepts)),
            [rf"$\mathit{{{concept}}}$" for concept in concept_names],
        )
        plt.grid(linewidth=0.3, zorder=1)
        if paper:
            plt.legend()
        else:
            plt.legend(bbox_to_anchor=(1.025, 1.0), loc="upper left", borderaxespad=0.0)
        # plt.legend()
        fig.tight_layout()
        plt.ylim(bottom=0, top=max(0.1, plt.ylim()[1]))
        plt.xlim(left=-0.5, right=0.5 + len(self.concepts) - 1)
        LOGGER.log_image(plt, self.concepts[0])
        plt.show()

    ##### Print stats and information #####
    def statsInfo(self):

        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not self.stats:
            raise Exception("Please let the model explain first")

        # Print a table with information
        table = PrettyTable(
            title=f"Model: {self.model.model_name}; Class: {self.target_class}; Examples: {self.test_images_folder}",
            field_names=["Concept", "Layer", "Attrib. mean", "Attrib. 95.45% CI"],
            float_format=".2",
        )
        for i, concept_name in enumerate(self.concepts):
            for j, layer_name in enumerate(self.layers):
                table.add_row(
                    [
                        concept_name if j == 0 else "",
                        layer_name,
                        f"{self.stats[layer_name][concept_name].mean:.3g} +- {self.stats[layer_name][concept_name].std:.3g}",
                        [
                            f"{self.stats[layer_name][concept_name].begin:.3g}",
                            f"{self.stats[layer_name][concept_name].end:.3g}",
                        ],
                    ],
                    # divider=True if j == len(self.layers)-1 else False,
                )
        LOGGER.log_table(table, self.concepts[0])
        print(table)

    ##### Function used to compute the FEATURE MAPS #####
    def computeFeatureMaps(self, layer_name):
        # Checks
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not layer_name:
            raise Exception("Please provide the function with one layer")

        # Compute the feature maps for each class
        self.model.activation_generator.concept_images_dir = self.test_images_dir
        class_feature_maps = (
            self.model.activation_generator.get_feature_maps_for_concept(
                self.test_images_folder, layer_name
            )
        )
        self.model.activation_generator.concept_images_dir = self.concept_images_dir

        return class_feature_maps


#####
# Model class
#####

class Model:

    ##### Init #####
    def __init__(
        self,
        model_name,
        graph_path_filename,
        label_path_filename,
        preprocessing_function=lambda x: x / 255,
        binary_classification=False,
        max_examples=500,
    ):
        # Attributes
        self.model_name = model_name
        self.max_examples = max_examples
        self.binary_classification = binary_classification

        # Folders and directories
        self.graph_path_filename = graph_path_filename
        self.label_path_filename = label_path_filename

        self.graph_path_dir = None
        self.label_path_dir = None

        # Wrapper & preprocessing functions
        self.model_wrapper = KerasModelWrapper
        self.activation_generator = ImageActivationGenerator
        self.preprocessing_function = preprocessing_function

    ##### Get layer names #####
    def getLayerNames(self):
        return [layer_name for layer_name in self.model_wrapper.layer_tensors.keys()]

    ##### Print model's informations #####
    def info(self):
        # Print a table with information
        table = PrettyTable(
            title=f"Model: {self.model_name}",
            field_names=["N. classes", "Layers"],
            float_format=".2",
        )
        for i, layer_name in enumerate(self.getLayerNames()):
            table.add_row(
                [len(self.model_wrapper.labels) if i == 0 else "", layer_name]
            )
        print(table)


#####
# ConceptLayer class
#####


class ConceptLayer:

    ##### Init #####
    def __init__(self):
        # Attributes
        self.attributions = {}
        self.concept_map = None

        # CAV
        self.cav = Cav()


#####
# Cav class
#####


class Cav:

    ##### Init #####
    def __init__(
        self, direction=None, centroid0=None, centroid1=None, concept_emblem=None
    ):
        # Attributes
        self.direction = direction
        self.centroid0 = centroid0
        self.centroid1 = centroid1
        self.concept_emblem = concept_emblem

    def __repr__(self):
        return f"CAV(direction={self.direction}, centroid0={self.centroid0}, centroid1={self.centroid1}, concept_emblem={self.concept_emblem})"


#####
# Prediction class
#####

class Prediction:

    ##### Init #####
    def __init__(self, class_name=None, class_index=None, confidence=None):
        # Attributes
        self.class_name = class_name
        self.class_index = class_index
        self.confidence = confidence


#####
# Predictions class
#####

class Predictions:

    ##### Init #####
    def __init__(self, predictions, test_image_filename, model_name):
        # Attributes
        self.predictions = predictions
        self.test_image_filename = test_image_filename
        self.model_name = model_name

    ##### Plot a table with the predictions information #####
    def info(self, num_of_classes=3):
        # Print a table with information
        table = PrettyTable(
            title=f"Model: {self.model_name}",
            field_names=["Image", "Class name", "Confidence"],
            float_format=".2",
        )
        for i in range(min(num_of_classes, len(self.predictions[0]))):
            table.add_row(
                [
                    self.test_image_filename if i == 0 else "",
                    self.predictions[0][i].class_name,
                    f"{self.predictions[0][i].confidence:.2g}",
                ]
            )
        print(table)


#####
# Stat class
#####

class Stat:

    ##### Init #####
    def __init__(self, attributions):
        # Attributes
        self.attributions = attributions

        # Simple
        self.mean, self.std = tf.reduce_mean(self.attributions), np.std(
            self.attributions
        )

        # Compute confidence interval
        # self.confidence = 0.95
        self.n = len(self.attributions)
        self.std_err = self.std / np.sqrt(self.n)
        # self.h = self.std_err * stats.t.ppf((1 + self.confidence) / 2, self.n-1)
        self.begin = tf.nn.relu(self.mean - self.std_err * 2)
        self.end = self.mean + self.std_err * 2


#####
# CustomColormap class
#####


class CustomColormap:
    ##### Init #####
    def __init__(self, nodes=None, colors=None, min=0, max=1, alpha=0.6):
        # Error handling
        if type(nodes) == type(colors):
            if hasattr(nodes, "__len__") and hasattr(nodes, "__len__"):
                if len(nodes) != len(colors):
                    raise ValueError("Arrays of different lengths")
            elif nodes is not None or colors is not None:
                raise ValueError("Type not supported")
        else:
            raise ValueError("Attributes of different types")
        if min >= max:
            raise ValueError
        # Set attributes
        self.nodes = nodes
        self.colors = colors
        self.min = min
        self.max = max
        self.alpha = alpha

    ##### Get LinearSegmentedColormap #####
    def getLinearSegmentedColormap(self):
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "custom", list(zip(self.nodes, self.colors))
        )

    ##### Plot imshow #####
    def imshow(self, heatmap):
        plt.imshow(
            heatmap,
            cmap=self.getLinearSegmentedColormap(),
            alpha=self.getAlpha(),
            vmin=self.getMin(),
            vmax=self.getMax(),
        )
        plt.clim(self.getMin(), self.getMax())

    # plt.colorbar(shrink=0.8)

    # Getters
    def getMin(self):
        return self.min

    def getMax(self):
        return self.max

    def getAlpha(self):
        return self.alpha


# Definition
original_colormap = cm.jet
colormap = CustomColormap(
    nodes=[0.0, 0.05] + [i for i in np.linspace(0.1, 1.0, 100)],
    colors=[(0, 0, 0, 1), (0, 0, 0, 1)]
    + [original_colormap(i) for i in np.linspace(0.15, 1.0, 100)],
)


#####
# KerasModelWrapper class
#####

class KerasModelWrapper:

    ##### Init #####
    def __init__(self, model_path, labels_path, batch_size):
        self.model_name = None  # Model name
        self.layers = []  # Layer names
        self.layer_tensors = None  # Tensors

        self.simulated_layer_model = {}  # Simulated "layer" model
        self.simulated_logits_model = {}  # Simulated "logits" model

        # Batching
        self.batch_size = batch_size

        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"mse": MeanSquaredError()}
        )

        # Convnext does not work on GPU on MacM1
        self.run_on_cpu = "convnext" in self.model.name and "Darwin" in platform.system()

        # Fetch tensors
        self._get_layer_tensors()
        # Load labels
        self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

    ##### Get the class label from its id #####
    def id_to_label(self, idx):
        return self.labels[idx]

    ##### Get the class id from its label #####
    def label_to_id(self, label):
        return self.labels.index(label)

    ##### Get the prediction(s) given one or more input(s) #####
    def get_predictions(self, imgs):

        # Feed the model with the inputs
        with tf.device('/CPU:0' if self.run_on_cpu else ""):
            inputs = tf.cast(imgs, tf.float32)
            predictions = self.model(inputs)
        # Return the predictions
        return predictions

    ##### Get the feature maps given one or more input(s) #####
    def get_feature_maps(self, imgs, layer_name):
        # Simulate a model with the selected layer as the last (lazy)
        if layer_name not in self.simulated_layer_model:
            self.simulated_layer_model[layer_name] = tf.keras.models.Model(
                inputs=[self.model.inputs], outputs=[self.layer_tensors[layer_name]]
            )

        # Compute the fmaps
        feature_maps = np.array([])
        for i in range(len(imgs)):
            q = i % self.batch_size
            if q == self.batch_size - 1 or i == len(imgs) - 1:
                inputs = tf.cast(imgs[i - q : min(i + 1, len(imgs))], tf.float32)
                with tf.device('/CPU:0' if self.run_on_cpu else ""):
                    output = self.simulated_layer_model[layer_name](inputs)
                if len(feature_maps) == 0:
                    feature_maps = output
                else:
                    feature_maps = np.concatenate((feature_maps, output))

        # Return the fmaps
        return feature_maps

    ##### Get the logits given a layer and one or more input(s) #####
    def get_logits(self, feature_maps, layer_name):
        # Simulate a model with the logits (lazy)
        if layer_name not in self.simulated_logits_model:
            self.simulated_logits_model[layer_name] = tf.keras.Model(
                inputs=self.layer_tensors[layer_name], outputs=self.model.outputs
            )
            self.simulated_logits_model[layer_name].layers[-1].activation = None

        # Feed the model with the inputs
        with tf.device('/CPU:0' if self.run_on_cpu else ""):
            logits = self.simulated_logits_model[layer_name](feature_maps)

        # Return the logits
        return logits

    ##### Get the gradients given a layer and one or more input(s) #####
    def get_gradient_of_score(self, feature_maps, layer_name, target_class_index):
        # Simulate a model with the logits (lazy)
        if layer_name not in self.simulated_logits_model:
            self.simulated_logits_model[layer_name] = tf.keras.Model(
                inputs=self.layer_tensors[layer_name], outputs=self.model.outputs
            )
            self.simulated_logits_model[layer_name].layers[-1].activation = None

        # Executing the gradients computation (batching)
        gradients = np.array([])
        for i in range(len(feature_maps)):
            q = i % self.batch_size
            if q == self.batch_size - 1 or i == len(feature_maps) - 1:
                inputs = tf.cast(
                    feature_maps[i - q : min(i + 1, len(feature_maps))], tf.float32
                )
                # Real batched computation
                with tf.device('/CPU:0' if self.run_on_cpu else ""):
                    with tf.GradientTape() as tape:
                        tape.watch(inputs)
                        logits = self.simulated_logits_model[layer_name](inputs)
                        logit = logits[..., target_class_index]
                    output = tape.gradient(logit, inputs)
                # Concatenating the batches' outputs
                if len(gradients) == 0:
                    gradients = output
                else:
                    gradients = np.concatenate((gradients, output))

        # Return the gradients
        return gradients

    ##### Get wrapped model's image shape #####
    def get_image_shape(self):
        input_shape = self.model.input_shape[1:]
        x = input_shape[0]
        y = input_shape[1]
        c = input_shape[2]
        return [x, y, c]

    # Util to get the layer tensors
    def _get_layer_tensors(self):
        self.layer_tensors = {}
        self.layers = self.model.layers
        self.model_name = self.model.name
        for layer in self.layers:
            # print(layer.name)
            # print(self.model_name)
            if "input" not in layer.name:
                # ResNet50V2
                if self.model_name == "resnet50v2":
                    if "conv4" in layer.name or "conv5" in layer.name:
                        if "_out" in layer.name:
                            self.layer_tensors[layer.name] = layer.output
                    elif layer.name == "post_relu":
                        self.layer_tensors[layer.name] = layer.output
                # VGG16
                elif self.model_name == "vgg16":
                    if (
                        "conv" in layer.name and "conv_1" not in layer.name
                    ):  # and "conv_2" not in layer.name:
                        self.layer_tensors[layer.name] = layer.output
                # InceptionV3
                elif self.model_name == "inception_v3":
                    if "mixed" in layer.name:
                        self.layer_tensors[layer.name] = layer.output
                else:
                    self.layer_tensors[layer.name] = layer.output


# Util to reshape the feature maps as needed to feed through the model network
# def reshape_feature_maps(self, layer_acts):
# 	return np.asarray(layer_acts).squeeze()

#####
# ImageActivationGenerator class
#####


class ImageActivationGenerator:
    ##### Init #####
    def __init__(
        self,
        model_wrapper,
        concept_images_dir,
        cache_dir,
        preprocessing_function=None,
        max_examples=500,
    ):
        self.model_wrapper = model_wrapper
        self.concept_images_dir = concept_images_dir
        self.cache_dir = cache_dir
        self.max_examples = max_examples
        self.preprocessing_function = preprocessing_function

    ##### Get feature maps for a concept #####
    def get_feature_maps_for_concept(self, concept, layer):
        images = self.get_images_for_concept(concept)
        if len(images) == 0:
            raise Exception("Please provide example images for each concept")
        feature_maps = self.model_wrapper.get_feature_maps(images, layer)
        return feature_maps

    ##### Compute or restore feature maps for all the [layers] and [concepts] #####
    def get_feature_maps_for_layers_and_concepts(
        self, layer_names, concepts, cache=True
    ):
        # Feature maps array
        feature_maps = {}
        # Initialize cache dir
        if self.cache_dir and not tf.io.gfile.exists(self.cache_dir):
            tf.io.gfile.makedirs(self.cache_dir)
        # For each concept
        for concept in concepts:
            if concept not in feature_maps:
                feature_maps[concept] = {}
            # For each layer
            for layer_name in layer_names:
                feature_maps_path = (
                    os.path.join(
                        self.cache_dir,
                        "f_maps_{}_{}.joblib".format(concept, layer_name),
                    )
                    if self.cache_dir
                    else None
                )
                if (
                    feature_maps_path
                    and tf.io.gfile.exists(feature_maps_path)
                    and cache
                ):
                    # Read from cache
                    feature_maps[concept][layer_name] = load(feature_maps_path)
                else:
                    # Compute and write to cache
                    feature_maps[concept][
                        layer_name
                    ] = self.get_feature_maps_for_concept(concept, layer_name)
                    if feature_maps_path and cache:
                        tf.io.gfile.mkdir(os.path.dirname(feature_maps_path))
                        dump(
                            feature_maps[concept][layer_name],
                            feature_maps_path,
                            compress=3,
                        )
        # Return the feature maps
        return feature_maps

    ##### Get the concept images from the concept folder #####
    def get_images_for_concept(self, concept, preprocess=True):
        # Construct filenames array
        concept_dir = os.path.join(self.concept_images_dir, concept)

        def is_image(filename):
            for ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
                if filename.lower().endswith(ext):
                    return True
            return False

        img_paths = [
            os.path.join(concept_dir, d)
            for d in tf.io.gfile.listdir(concept_dir)
            if is_image(d)
        ]
        # Load the images with the filenames
        imgs = self._load_images_from_files(
            img_paths,
            self.max_examples,
            shape=self.model_wrapper.get_image_shape()[:2],
            preprocess=preprocess,
        )
        # Return the loaded images
        return imgs

    # Util that, given some filenames, loads the images
    def _load_images_from_files(
        self, filenames, max_imgs=500, shape=(224, 224), preprocess=True
    ):
        # Images array
        imgs = []
        # Load all the images in parallel
        pool = multiprocessing.Pool(50)  # Run the parallel algorithm with 50 workers
        imgs = pool.map(
            lambda filename: self._load_image_from_file(
                filename, shape, preprocess=preprocess
            ),
            filenames[:max_imgs],
        )
        pool.close()
        imgs = [img for img in imgs if img is not None]
        # Return the images as an np array
        return np.array(imgs)

    # Util that, given a filename, loads an image
    def _load_image_from_file(self, filename, shape, preprocess=True):
        try:
            img = np.array(
                PIL.Image.open(tf.io.gfile.GFile(filename, "rb"))
                .convert("RGB")
                .resize(shape, PIL.Image.BILINEAR),
                # dtype=np.float32
            )
        except:
            return None
        if self.preprocessing_function is not None and preprocess:
            img = self.preprocessing_function(img)
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            return None
        else:
            return img
