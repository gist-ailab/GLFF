from net_hard import GlobalLocalLesionClassifier
from rein import ReinsDinoVisionTransformer
from util.training import set_requires_grad


def create_glff_model(variant, state_dict, num_classes, device):
    global_encoder = ReinsDinoVisionTransformer(**variant)
    local_encoder = ReinsDinoVisionTransformer(**variant)
    global_encoder.load_state_dict(state_dict, strict=False)
    local_encoder.load_state_dict(state_dict, strict=False)

    model = GlobalLocalLesionClassifier(
        global_encoder=global_encoder,
        local_encoder=local_encoder,
        embed_dim=variant["embed_dim"],
        num_classes=num_classes,
        patch_size=variant["patch_size"],
        tau=0.0,
        min_tokens=4,
        num_heads=8,
        cross_attn_layers=1,
        fusion="concat",
        hidden_dim=512,
        cross_attn_warmup_epochs=5,
        use_global_cls=True,
    )
    set_requires_grad(model, ["reins", "head", "cross_attn", "global_encoder", "local_encoder"])
    return model.to(device)
