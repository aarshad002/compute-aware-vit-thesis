import timm
from models.vit_static import build_static_model
from models.vit_dynamic import build_dynamic_model


def build_model(config):
    model_type = config["model"].get("type", "dense")

    if model_type == "dense":
        model_name = config["model"]["name"]
        num_classes = config["model"]["num_classes"]
        pretrained = config["model"].get("pretrained", True)

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model

    elif model_type == "static":
        return build_static_model(config)

    elif model_type == "dynamic":
        return build_dynamic_model(config)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")