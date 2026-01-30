# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.19.7",
#     "pillow==12.1.0",
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


@app.cell(column=1)
def _():
    import marimo as mo
    from PIL import Image
    import torch
    from torchvision import models, transforms
    import torch.nn.functional as F
    return F, Image, mo, models, transforms


@app.cell
def _(Image):
    import glob
    import random
    import os

    random.seed(42)

    image_paths = random.sample(glob.glob('images/*.JPEG'), 10)

    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    images = [get_image(path) for path in image_paths]
    images
    return images, os


@app.cell
def _(images, transforms):
    def transform_image(img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])       
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])    

        return transf(img).unsqueeze(0)

    trans_images = [transform_image(img) for img in images]
    trans_images
    return (trans_images,)


@app.cell
def _(os):
    import json

    idx2label, cls2label, cls2idx = [], {}, {}
    with open(os.path.abspath('./imagenet_class_index.json'), 'r') as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}    

    cls2label
    return (idx2label,)


@app.cell(column=2)
def _(models):
    model = models.inception_v3(pretrained=True)
    return (model,)


@app.cell
def _(model, trans_images):
    model.eval()
    logits = [model(img_t) for img_t in trans_images]
    logits
    return (logits,)


@app.cell
def _(F, logits):
    probs = [F.softmax(logit, dim=1).topk(5) for logit in logits]
    probs
    return (probs,)


@app.cell
def _(idx2label, images, mo, probs):
    mo.vstack(
        [mo.hstack([img, "\n".join([idx2label[label] for label in classes])]) for img, classes in zip(images, probs)]
    )
    return


if __name__ == "__main__":
    app.run()
