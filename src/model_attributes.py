model_attributes = {
    "bert": {
        "feature_type": "text"
    },
    "inception_v3": {
        "feature_type": "image",
        "target_resolution": (299, 299),
        "flatten": False,
    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "last_layer_name": 'fc',
        "feature_filename": 'resnet50_features.npy',
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "linear": {
        "feature_filename": 'resnet50_features.npy',
        "feature_type": "precomputed",
    },
    "resnet34": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": False
    },
    "raw_logistic_regression": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": True,
    },
    "bert-base-uncased": {
        'feature_type': 'text'
    },
}