# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lime==0.2.0.1",
#     "marimo>=0.19.7",
#     "matplotlib==3.10.8",
#     "numpy==2.2.6",
#     "pillow==12.1.0",
#     "scikit-image==0.25.2",
#     "torch==2.10.0",
#     "torchvision==0.25.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="columns", css_file="", auto_download=["html"])


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # **TP : Explicabilité des modèles de deep learning pour les images avec LIME et SHAP**

    ## **Objectif**
    Ce TP vise à explorer comment interpréter les prédictions d’un modèle de deep learning appliqué aux images à l’aide de **LIME** et **SHAP**. Ces outils permettent d’identifier les parties de l’image qui influencent le plus les décisions du modèle, contribuant ainsi à rendre les modèles plus transparents et compréhensibles.

    ---

    ## **Contexte**
    Les modèles de deep learning, bien qu’efficaces, sont souvent considérés comme des boîtes noires. Comprendre pourquoi un modèle fait une certaine prédiction est essentiel pour :
    1. Identifier les biais potentiels dans le modèle.
    2. Valider la fiabilité des prédictions dans des applications sensibles.
    3. Construire la confiance des utilisateurs finaux.

    Dans ce TP, nous allons :
    1. Charger un modèle de classification d’images pré-entraîné avec PyTorch.
    2. Utiliser ce modèle pour prédire les classes d’images fournies.
    3. Appliquer **LIME** et **SHAP** pour expliquer ces prédictions.
    4. Comparer les visualisations générées par ces deux outils et analyser les résultats.

    ---

    ## **Plan du TP**

    ### **Étape 1 : Préparation de l’environnement**
    1. Installer les bibliothèques nécessaires pour PyTorch, LIME, et SHAP.
    2. Télécharger ou préparer un ensemble d’images pour les tests (votre dataset préféré, des images issues d’ImageNet, etc.).

    ---

    ### **Étape 2 : Charger un modèle pré-entraîné**
    1. Nous allons utiliser un modèle pré-entraîné disponible dans PyTorch, comme ResNet18 ou VGG16, avec des poids pré-entraînés sur ImageNet.
    2. Le modèle sera utilisé pour effectuer des prédictions sur les images, après les avoir redimensionnées et normalisées en fonction des besoins du modèle.
    3. Pour chaque image, nous afficherons les classes prédites avec leurs scores de confiance.

    ---

    ### **Étape 3 : Explicabilité avec LIME**
    1. **Présentation de LIME :**
       LIME fonctionne en créant des perturbations localisées sur une image (par exemple, en masquant des zones spécifiques) pour mesurer l’impact de chaque zone sur la prédiction.
    2. **Étapes :**
       - Segmenter l’image en pixels ou en superpixels.
       - Identifier les zones les plus influentes pour une prédiction donnée.
       - Générer une heatmap mettant en évidence les zones importantes pour la classe prédite.
    3. Nous analyserons les résultats pour comprendre quelles parties de l’image influencent le plus la prédiction.

    ---

    ### **Étape 4 : Explicabilité avec SHAP**
    1. **Présentation de SHAP :**
       SHAP utilise la théorie des jeux pour attribuer une importance à chaque pixel ou groupe de pixels, en mesurant leur contribution à la prédiction.
    2. **Étapes :**
       - Fournir les images prétraitées au modèle pour calculer les valeurs SHAP.
       - Générer une visualisation qui montre les contributions positives et négatives des pixels ou des superpixels pour la prédiction.
    3. Nous interpréterons les visualisations en observant les zones qui favorisent ou défavorisent la classe prédite.

    ---

    ### **Étape 5 : Comparaison et analyse des résultats**
    1. **Comparaison des visualisations :**
       - Identifier les différences et similitudes entre les résultats de LIME et SHAP.
       - Analyser les zones mises en évidence par chaque méthode.
    2. **Discussion :**
       - Quels sont les points forts et limites de chaque méthode ?
       - Les deux approches donnent-elles des explications cohérentes ?
       - Quels défis rencontrons-nous en utilisant ces outils avec des modèles de grande taille ou des images complexes ?

    ---

    ## **Livrables attendus**
    1. Les heatmaps générées par LIME et SHAP pour au moins deux images.
    2. Une analyse comparative des résultats obtenus avec LIME et SHAP.
    ---
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the data
    """)
    return


@app.cell
def _():
    import marimo as mo
    from PIL import Image
    import torch
    from torchvision import models, transforms
    import torch.nn.functional as F
    import glob
    import random
    import os, json
    return F, Image, glob, json, mo, models, os, torch, transforms


@app.cell
def _(Image, os, transforms):
    def get_image(path):
        with open(os.path.abspath(path), "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def resize_image(img):
        transf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
            ]
        )

        return transf(img)


    def process_image(img):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transf = transforms.Compose([transforms.ToTensor(), normalize])

        return transf(img)


    def transform_image(img):
        return process_image(resize_image(img))
    return get_image, process_image, resize_image, transform_image


@app.cell
def _(models, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="DEFAULT")
    return device, model


@app.cell
def _(F, get_image, glob, model, transform_image):
    model.to('cpu')
    model.eval()
    probs_paths = [(F.softmax(model(transform_image(get_image(path)).unsqueeze(0)), dim=1), path) for path in glob.glob('./images/*.JPEG')]
    return (probs_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## In order to find some useful samples, let's choose the ones where the model was hesitating the most between the 2 top labels.
    """)
    return


@app.cell
def _(probs_paths):
    def get_margin(probs):
        top = probs.topk(2)
        diff = top[0][0][0] - top[0][0][1]
        return diff if top[0][0][0] > 0.4 else 1

    margins = [get_margin(probs) for probs, _ in probs_paths]

    paths = [path[1] for m, path in zip(margins, probs_paths) if m < 0.08]
    paths
    return (paths,)


@app.cell
def _(get_image, paths):
    images = [get_image(path) for path in paths[:15]]
    images
    return (images,)


@app.cell
def _(images, transform_image):
    trans_images = [transform_image(img).unsqueeze(0) for img in images]
    return (trans_images,)


@app.cell
def _(json, os):
    idx2label, cls2label, cls2idx = [], {}, {}
    with open(os.path.abspath("./imagenet_class_index.json"), "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {
            class_idx[str(k)][0]: class_idx[str(k)][1]
            for k in range(len(class_idx))
        }
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    cls2label
    return (idx2label,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's check what are the top predicted labels for the chosen samples
    """)
    return


@app.cell
def _(device, model, trans_images):
    model.to(device)
    model.eval()
    logits = [model(img_t.to(device)) for img_t in trans_images]
    logits
    return (logits,)


@app.cell
def _(F, logits):
    probs = [F.softmax(logit, dim=1).topk(5) for logit in logits]
    probs
    return (probs,)


@app.cell
def _(idx2label, probs):
    prob_label = [
        [
            (p, idx2label[idx])
            for p, idx in zip(
                classes[0][0].detach().cpu().numpy(),
                classes[1][0].detach().cpu().numpy(),
            )
        ]
        for classes in probs
    ]
    prob_label
    return (prob_label,)


@app.cell
def _(images, mo, prob_label):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.center(img),
                    mo.ui.table(
                        data=[
                            {"proba": f"{elem[0]:.4f}", "label": elem[1]}
                            for elem in classes
                        ],
                        selection=None,
                    ),
                ],
                widths=[2, 1],
                justify="center",
                align="center",
            )
            for img, classes in zip(images, prob_label)
        ]
    )
    return


@app.cell(column=3)
def _():
    from lime import lime_image
    import numpy as np
    from skimage.segmentation import mark_boundaries
    import matplotlib.pyplot as plt
    return lime_image, mark_boundaries, np


@app.cell
def _(F, device, model, process_image, torch):
    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple([process_image(img) for img in images]), dim=0)

        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
    return (batch_predict,)


@app.cell
def _(
    Image,
    batch_predict,
    idx2label,
    lime_image,
    mark_boundaries,
    np,
    resize_image,
):
    def explain_image(
        img, pos_only=False, num_features=10, hide_rest=True, top_labels=5
    ):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(resize_image(img)),
            batch_predict,
            top_labels=top_labels,
            hide_color=0,
            num_samples=1000,
        )
        boundaries = []
        for label in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=pos_only,
                num_features=num_features,
                hide_rest=hide_rest,
            )
            img_boundry = mark_boundaries(temp / 255.0, mask)
            boundaries.append(
                (
                    Image.fromarray((img_boundry * 255).astype("uint8"), "RGB"),
                    idx2label[label],
                )
            )
        return boundaries
    return (explain_image,)


@app.cell
def _(explain_image, images):
    exps = [
        [(img, "")]
        + explain_image(img, num_features=5, pos_only=True, top_labels=3)
        for img in images
    ]
    return (exps,)


@app.cell
def _(exps, mo):
    mo.vstack(
        [
            mo.hstack(
                [mo.image(img, caption=cap) for img, cap in exp],
                justify="center",
                align="center",
            )
            for exp in exps
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## In examples such as the firetruck and mushroom pictures, we can see clearly which parts of the image caused the model to swing toward one label or another. In other examples the differences are more subtle but we notice that in sax, for example, it is the particular parts of the instrument that can cause the model to identify it as a basson or sax.
    """)
    return


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
