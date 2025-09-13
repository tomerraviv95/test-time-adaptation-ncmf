import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from utils.misc import ema_update_model
from augmentations.transforms_cotta import get_tta_transforms

from models.kalman_gain_net import GainNet
from utils.knet_feats import collect_knet_feats


@ADAPTATION_REGISTRY.register()
class NCMF(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.class_probs_ema = (1 / self.num_classes) * torch.ones(
            self.num_classes, device=self.device
        )
        self.tta_transform = get_tta_transforms(
            self.img_size, padding_mode="reflect", cotta_augs=False
        )

        self.sce = SymmetricCrossEntropy()
        self.slr = SoftLikelihoodRatio()
        self.ent = Entropy()

        # source + hidden
        self.src_model = deepcopy(self.model)
        for p in self.src_model.parameters():
            p.detach_()
        self.hidden_model = deepcopy(self.model)
        for p in self.hidden_model.parameters():
            p.detach_()

        # filter scaffolding
        self.alpha = cfg.CMF.ALPHA
        self.gamma = cfg.CMF.GAMMA
        self.post_type = cfg.CMF.TYPE
        self.hidden_var = torch.tensor(0.0, device=self.device)
        param_size_ratio = self.num_trainable_params / 38400
        self.q = cfg.CMF.Q * param_size_ratio

        # learned gain K in [min_gain, max_gain]
        self.min_gain = cfg.KNET.MIN_GAIN
        self.max_gain = cfg.KNET.MAX_GAIN
        self.smooth = cfg.KNET.SMOOTH
        in_dim = len(cfg.KNET.FEATS)
        self.gain_net = GainNet(
            in_dim=in_dim,
            hidden_dim=cfg.KNET.HIDDEN_DIM,
            per_layer=cfg.KNET.PER_LAYER,
            n_layers=1,
        ).to(self.device)
        self.pred_gain_ema = torch.tensor(0.0, device=self.device)
        if cfg.KNET.CHECKPOINT:
            self.gain_net.load_state_dict(
                torch.load(cfg.KNET.CHECKPOINT, map_location="cpu")
            )

        # for reset
        self.models = [self.src_model, self.model, self.hidden_model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def collect_params(self):
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        # (optional) sample weighting/consistency like CMF could be added; start simple
        slr_vals = self.slr(logits=outputs).detach()
        ent_vals = self.ent(logits=outputs).detach()
        loss = slr_vals.sum() / self.batch_size
        return outputs, loss, slr_vals, ent_vals

    @torch.no_grad()
    def knet_gain(self, outputs, slr_vals, ent_vals, grad_norm):
        feats = collect_knet_feats(
            outputs,
            slr_vals,
            ent_vals,
            self.model,
            self.hidden_model,
            self.src_model,
            grad_norm=grad_norm,
            batch_size=self.batch_size,
            img_size=max(self.img_size),
        )
        pred = self.gain_net(feats).squeeze()
        K = self.min_gain + (self.max_gain - self.min_gain) * pred
        self.pred_gain_ema = self.smooth * self.pred_gain_ema + (1 - self.smooth) * K
        return self.pred_gain_ema.clamp(self.min_gain, self.max_gain)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, slr_vals, ent_vals = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            grad_norm = torch.tensor(
                0.0, device=self.device
            )  # skip grad norm in AMP path for simplicity
        else:
            outputs, loss, slr_vals, ent_vals = self.loss_calculation(x)
            loss.backward()
            # mean abs grad over adapted params
            grads = [
                p.grad.detach().flatten().abs().mean()
                for p in self.params
                if p.grad is not None
            ]
            grad_norm = (
                torch.stack(grads).mean()
                if len(grads) > 0
                else torch.tensor(0.0, device=self.device)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            # predict/recover hidden toward source
            recovered = ema_update_model(
                self.hidden_model,
                self.src_model,
                self.alpha,
                self.device,
                update_all=True,
            )

            # variance predict (for potential future use/logging)
            self.hidden_var = (self.alpha**2) * self.hidden_var + self.q

            # learned measurement fusion with gain K
            K = self.knet_gain(outputs, slr_vals, ent_vals, grad_norm)

            for ph, pr, pl in zip(
                self.hidden_model.parameters(),
                recovered.parameters(),
                self.model.parameters(),
            ):
                ph.data = (1 - K) * pr.data + K * pl.data

            # ensemble filtered state back to live
            merged_from = recovered if self.post_type == "op" else self.hidden_model
            self.model = ema_update_model(
                self.model, merged_from, self.gamma, self.device
            )

            # optional prior correction identical to CMF
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(
                    prior
                )
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        return outputs
