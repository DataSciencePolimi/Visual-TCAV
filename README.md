# Visual-TCAV

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Welcome to the official TensorFlow V2 implementation of Visual-TCAV, a novel framework for post-hoc explainability in image classification.

## Get started with Visual-TCAV

In this repository you will find all the necessary items to run the method:

| File/folder        | Description                                                                    |
|:------------------ |:------------------------------------------------------------------------------ |
| VisualTCAV_guide.ipynb      | Detailed guide on how to install and run Visual-TCAV |
| VisualTCAV.py | Visual-TCAV code                                           |
| datasets_and_models_downloader        | Code and files to automatically download the datasets and the models                           |

## Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification

#### Antonio De Santis, Riccardo Campi, Matteo Bianchi, Marco Brambilla

Convolutional Neural Networks (CNNs) have seen significant performance improvements in recent years. However, due to their size and complexity, they function as black-boxes, leading to transparency concerns. State-of-the-art saliency methods generate local explanations that highlight the area in the input image where a class is identified but cannot explain how a concept of interest contributes to the prediction, which is essential for bias mitigation. On the other hand, concept-based methods, such as TCAV (Testing with Concept Activation Vectors), provide insights into how sensitive is the network to a concept, but cannot compute its attribution in a specific prediction nor show its location within the input image. This paper introduces a novel post-hoc explainability framework, Visual-TCAV, which aims to bridge the gap between these methods by providing both local and global explanations for CNN-based image classification. Visual-TCAV uses Concept Activation Vectors (CAVs) to generate saliency maps that show where concepts are recognized by the network. Moreover, it can estimate the attribution of these concepts to the output of any class using a generalization of Integrated Gradients. This framework is evaluated on popular CNN architectures, with its validity further confirmed via experiments where ground truth for explanations is known, and a comparison with TCAV.

For more information, please refer to the paper available <a href="https://arxiv.org/abs/2411.05698">here</a>.

### Citation

If you find our work useful, please cite:

```
@misc{visualtcav,
      title={Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification}, 
      author={Antonio De Santis and Riccardo Campi and Matteo Bianchi and Marco Brambilla},
      year={2025},
      eprint={2411.05698},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.05698}, 
}
```
