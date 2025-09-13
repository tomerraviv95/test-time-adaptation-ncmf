import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_param_energy(a, b):
    vals = []
    for pa, pb in zip(a.parameters(), b.parameters()):
        vals.append(F.cosine_similarity(pa.flatten(), pb.flatten(), dim=-1))
    return torch.stack(vals).mean()


@torch.no_grad()
def param_l2_delta(a, b):
    num = torch.tensor(0.0, device=next(a.parameters()).device)
    den = torch.tensor(0.0, device=num.device)
    for pa, pb in zip(a.parameters(), b.parameters()):
        da = (pa - pb).float()
        num += (da * da).sum()
        den += (pb.float() * pb.float()).sum() + 1e-8
    return (num / den).sqrt()


@torch.no_grad()
def collect_knet_feats(
    outputs,
    slr_values,
    ent_values,
    model_live,
    model_hidden,
    model_src,
    grad_norm=0.0,
    batch_size=0,
    img_size=224,
):
    feats = []
    feats += [slr_values.mean(), slr_values.std()]
    feats += [ent_values.mean(), ent_values.std()]
    feats += [cosine_param_energy(model_src, model_live)]
    feats += [cosine_param_energy(model_src, model_hidden)]
    feats += [param_l2_delta(model_live, model_hidden)]
    if not isinstance(grad_norm, torch.Tensor):
        grad_norm = torch.tensor(float(grad_norm), device=outputs.device)
    feats += [grad_norm]
    feats += [float(batch_size), float(img_size)]
    return torch.stack(
        [torch.tensor(f, device=outputs.device).float() for f in feats], dim=0
    ).view(1, -1)
