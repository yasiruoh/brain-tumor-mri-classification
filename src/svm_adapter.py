"""Optional fitted SVM on the exact 2048-D CNN embedding. No SVM is bundled."""
from pathlib import Path
from src.engine import load_model, transform_image

def predict_svm(image, artifact="models/svm.skops"):
    import torch
    import skops.io as sio
    path = Path(artifact)
    if not path.is_file():
        raise FileNotFoundError("Optional SVM artifact is not installed. CNN inference remains available.")
    unknown = sio.get_untrusted_types(file=path)
    if unknown:
        raise ValueError("SVM contains unapproved types; refusing to deserialize.")
    pipeline = sio.load(path, trusted=[])
    backbone = load_model().backbone
    with torch.inference_mode():
        x = transform_image(image)
        for layer in list(backbone.children())[:-1]:
            x = layer(x)
        features = x.flatten(1).numpy()
    if getattr(pipeline, "n_features_in_", None) != 2048:
        raise ValueError("SVM must accept the exact 2048-D CNN embedding.")
    return {"label": str(pipeline.predict(features)[0]), "backend": "CNN embedding + SVM"}
