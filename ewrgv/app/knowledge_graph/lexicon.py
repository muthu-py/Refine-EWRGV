"""
app/knowledge_graph/lexicon.py
--------------------------------
Seed vocabularies for entity tagging.

Each entry is ``canonical name -> list of surface forms``.  Surface forms are
matched case-insensitively on word boundaries (see tagger.py); the canonical
name is what the knowledge graph stores.  Acronym forms that are ambiguous in
lower case (e.g. "SVM" is fine, "RF" is not) are marked with a leading "=",
meaning "match case-sensitively".

The per-run query understanding (methods, datasets, metrics, entities,
terminology/method variants) is added on top of these at build time, so the
lexicon only needs general ML / evaluation vocabulary plus domain packs.
Extend a pack, or add a new one, when working in another domain.
"""

from __future__ import annotations

METHOD: dict[str, list[str]] = {
    "convolutional neural network": ["convolutional neural network", "convolutional neural networks", "=CNN", "=CNNs", "convnet"],
    "vision transformer": ["vision transformer", "vision transformers", "=ViT", "=ViTs"],
    "swin transformer": ["swin transformer", "swin"],
    "transformer": ["transformer", "transformers", "self-attention"],
    "attention mechanism": ["attention mechanism", "attention mechanisms", "attention module", "attention-based"],
    "u-net": ["u-net", "unet", "u net"],
    "resnet": ["resnet", "resnet50", "resnet-50", "resnet101", "residual network"],
    "densenet": ["densenet", "densenet121", "densenet-121"],
    "efficientnet": ["efficientnet", "efficientnetb0", "efficientnet-b0"],
    "inception": ["inception", "inceptionv3", "inception-v3", "xception"],
    "vgg": ["vgg", "vgg16", "vgg-16", "vgg19"],
    "mobilenet": ["mobilenet", "mobilenetv2", "mobilenet-v2"],
    "recurrent neural network": ["recurrent neural network", "=RNN", "=LSTM", "=BiLSTM", "long short-term memory", "=GRU", "gated recurrent unit"],
    "generative adversarial network": ["generative adversarial network", "generative adversarial networks", "=GAN", "=GANs"],
    "graph neural network": ["graph neural network", "graph neural networks", "=GNN", "=GNNs", "graph convolutional network"],
    "support vector machine": ["support vector machine", "support vector machines", "=SVM", "=SVMs"],
    "random forest": ["random forest", "random forests"],
    "gradient boosting": ["gradient boosting", "xgboost", "lightgbm", "catboost", "=GBM"],
    "logistic regression": ["logistic regression"],
    "k-nearest neighbours": ["k-nearest neighbour", "k-nearest neighbor", "=KNN", "=k-NN"],
    "decision tree": ["decision tree", "decision trees"],
    "naive bayes": ["naive bayes"],
    "bayesian neural network": ["bayesian neural network", "bayesian convolutional neural network", "=BCNN", "=BNN", "monte carlo dropout"],
    "transfer learning": ["transfer learning", "pretrained model", "pre-trained model", "fine-tuning", "fine tuning"],
    "semi-supervised learning": ["semi-supervised learning", "semi-supervised", "=SSL"],
    "self-supervised learning": ["self-supervised learning", "self-supervised", "contrastive learning"],
    "federated learning": ["federated learning"],
    "ensemble learning": ["ensemble learning", "ensemble model", "ensemble models", "ensemble method"],
    "explainable ai": ["explainable ai", "explainable artificial intelligence", "=XAI", "interpretability", "explainability", "grad-cam", "gradcam", "saliency map", "=SHAP", "=LIME"],
    "data augmentation": ["data augmentation", "augmentation"],
    "multimodal learning": ["multimodal learning", "multimodal fusion", "multi-modal", "multimodal"],
    "feature fusion": ["feature fusion", "feature-level fusion", "feature level fusion"],
    "segmentation network": ["segmentation network", "semantic segmentation"],
    "local binary pattern": ["local binary pattern", "local binary patterns", "=LBP"],
    "image enhancement": ["image enhancement", "=CLAHE", "contrast limited adaptive histogram equalization", "illumination normalization", "illumination normalisation"],
    "foundation model": ["foundation model", "foundation models"],
    "large language model": ["large language model", "large language models", "=LLM", "=LLMs"],
    "mamba": ["mamba", "vision mamba", "state space model"],
    "squeeze and excitation": ["squeeze and excitation", "squeeze-and-excitation", "=SE block", "=SE blocks"],
    "uncertainty estimation": ["uncertainty estimation", "uncertainty quantification", "uncertainty estimates", "predictive uncertainty"],
    "knowledge distillation": ["knowledge distillation"],
    "edge deployment": ["edge ai", "edge computing", "edge device", "edge devices", "on-device"],
}

TASK: dict[str, list[str]] = {
    "classification": ["classification", "classifier", "classifying"],
    "multi-label classification": ["multi-label classification", "multilabel classification", "multi-label"],
    "multi-disease classification": ["multi-disease", "multidisease", "multiple diseases", "multi-class disease"],
    "segmentation": ["segmentation", "segmenting"],
    "detection": ["detection", "detecting"],
    "screening": ["screening"],
    "grading": ["grading", "severity grading", "severity assessment", "staging"],
    "diagnosis": ["diagnosis", "diagnostic", "computer-aided diagnosis", "=CAD"],
    "prediction": ["prediction", "predicting", "prognosis"],
    "localization": ["localization", "localisation", "lesion localization"],
    "real-time inference": ["real-time", "real time"],
}

METRIC: dict[str, list[str]] = {
    "accuracy": ["accuracy"],
    "auc": ["=AUC", "=AUROC", "area under the curve", "roc curve"],
    "f1-score": ["f1-score", "f1 score", "=F1", "macro-f1", "macro f1"],
    "sensitivity": ["sensitivity", "recall"],
    "specificity": ["specificity"],
    "precision": ["precision"],
    "cohen kappa": ["kappa", "cohen's kappa", "quadratic weighted kappa"],
    "dice coefficient": ["dice", "dice coefficient", "dice score"],
    "iou": ["=IoU", "intersection over union", "jaccard index"],
    "hamming loss": ["hamming loss"],
    "computational cost": ["computational cost", "computational complexity", "inference time", "latency", "computational requirements"],
}

# ---- Domain pack: ophthalmic / medical imaging ---------------------------- #

MODALITY: dict[str, list[str]] = {
    "fundus photography": ["fundus", "fundus image", "fundus images", "fundus photograph", "fundus photographs", "retinal fundus", "color fundus", "=CFP"],
    "optical coherence tomography": ["optical coherence tomography", "=OCT", "=SD-OCT", "oct scan", "oct scans", "oct images"],
    "oct angiography": ["oct angiography", "=OCTA", "=OCT-A"],
    "ultra-widefield imaging": ["ultra-widefield", "ultra-wide-field", "ultra wide field", "=UWF", "=UFI", "=UFIs"],
    "fluorescein angiography": ["fluorescein angiography", "=FFA", "=FA"],
    "slit lamp": ["slit lamp", "slit-lamp"],
    "clinical data": ["clinical data", "electronic health record", "electronic health records", "=EHR", "patient metadata"],
    "smartphone imaging": ["smartphone", "mobile phone camera", "handheld camera"],
}

DISEASE: dict[str, list[str]] = {
    "diabetic retinopathy": ["diabetic retinopathy", "=DR"],
    "glaucoma": ["glaucoma"],
    "age-related macular degeneration": ["age-related macular degeneration", "macular degeneration", "=AMD", "wet amd", "dry amd"],
    "diabetic macular edema": ["diabetic macular edema", "diabetic macular oedema", "=DME", "macular edema", "macular oedema"],
    "cataract": ["cataract", "cataracts"],
    "choroidal neovascularization": ["choroidal neovascularization", "=CNV"],
    "drusen": ["drusen"],
    "central serous retinopathy": ["central serous retinopathy", "central serous chorioretinopathy", "=CSR", "=CSCR"],
    "retinal vein occlusion": ["retinal vein occlusion", "=RVO"],
    "hypertensive retinopathy": ["hypertensive retinopathy"],
    "myopia": ["myopia", "refractive error", "refractive errors"],
    "retinopathy of prematurity": ["retinopathy of prematurity", "=ROP"],
    "alzheimer's disease": ["alzheimer", "alzheimer's disease"],
    "parkinson's disease": ["parkinson", "parkinson's disease"],
    "multiple sclerosis": ["multiple sclerosis"],
}

DATASET: dict[str, list[str]] = {
    "eyepacs": ["eyepacs", "kaggle dr"],
    "aptos": ["aptos", "aptos 2019"],
    "messidor": ["messidor", "messidor-2"],
    "drive": ["=DRIVE"],
    "stare": ["=STARE"],
    "chase_db1": ["chase_db1", "chase db1", "=CHASE"],
    "idrid": ["idrid"],
    "odir": ["odir", "odir-5k"],
    "rfmid": ["rfmid"],
    "kermany oct": ["kermany", "uscd oct", "ucsd"],
    "duke oct": ["duke dataset", "duke oct", "duke"],
    "origa": ["origa"],
    "refuge": ["refuge"],
    "biomisa": ["biomisa"],
    "fgadr": ["fgadr"],
    "ddr": ["=DDR"],
}

POPULATION: dict[str, list[str]] = {
    "paediatric patients": ["children", "paediatric", "pediatric", "infants", "neonates"],
    "elderly patients": ["elderly", "older adults", "aged population"],
    "low-resource settings": ["low-resource", "resource-constrained", "resource constrained", "underserved", "rural", "low- and middle-income", "developing countries"],
    "diverse populations": ["diverse populations", "multi-ethnic", "ethnicities", "ethnic groups", "demographic"],
    "multi-centre cohorts": ["multi-centre", "multi-center", "multicentre", "multicenter", "external validation", "external test"],
}

# Generic problem concepts that gap statements are usually about.
CONCEPT: dict[str, list[str]] = {
    "generalization": ["generalization", "generalisation", "generalize", "generalise", "cross-dataset", "domain shift", "domain adaptation"],
    "class imbalance": ["class imbalance", "imbalanced data", "imbalanced dataset", "imbalanced datasets", "data imbalance"],
    "labelled data scarcity": ["labeled data", "labelled data", "annotated data", "annotation cost", "limited data", "small dataset", "small datasets", "data scarcity"],
    "image quality variation": ["image quality", "illumination", "low-quality", "noise", "artifacts", "artefacts"],
    "clinical deployment": ["clinical deployment", "clinical practice", "clinical settings", "clinical use", "real-world deployment", "clinical applicability", "clinical validation"],
    "robustness": ["robustness", "robust"],
    "uncertainty": ["uncertainty", "reliability", "confidence estimates"],
    "interpretability": ["interpretability", "interpretable", "transparency", "trust"],
    "small lesion detection": ["small lesion", "small lesions", "microaneurysm", "microaneurysms", "subtle lesions"],
    "privacy": ["privacy", "data sharing"],
}

ENTITY_TYPES: dict[str, dict[str, list[str]]] = {
    "METHOD": METHOD,
    "TASK": TASK,
    "METRIC": METRIC,
    "MODALITY": MODALITY,
    "DISEASE": DISEASE,
    "DATASET": DATASET,
    "POPULATION": POPULATION,
    "CONCEPT": CONCEPT,
}
