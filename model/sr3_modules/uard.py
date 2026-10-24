# Maintained for the UARD source preview. See THIRD_PARTY_NOTICES.md
# and docs/SOURCE_CHANGES.md for provenance and release modifications.
"""Optical-domain residual diffusion with SAR-conditioned noise scale.

Laplace s is log(scale), not log(variance). No optical target is used at inference.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class OpticalAnchor(nn.Module):
    def __init__(self, in_channels, out_channels=3, width=32,
                 log_scale_min=-5.0, log_scale_max=1.0):
        super().__init__()
        self.lower, self.upper = log_scale_min, log_scale_max
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width, 3, padding=1), nn.SiLU())
        self.anchor_head = nn.Conv2d(width, out_channels, 3, padding=1)
        self.scale_head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, sar):
        features = self.encoder(sar)
        features = F.interpolate(features, size=sar.shape[-2:], mode='bilinear', align_corners=False)
        anchor = self.anchor_head(features).tanh()
        s = self.lower + (self.upper - self.lower) * self.scale_head(features).sigmoid()
        # Fixed monotone normalization of the Laplace scale; never per-image min/max.
        uncertainty = (s.exp() - math.exp(self.lower)) / (math.exp(self.upper) - math.exp(self.lower))
        return anchor, s, uncertainty


class UARDiffusion(nn.Module):
    MODES = ('full', 'residual_uniform', 'anchor_condition', 'uncertainty_only')

    def __init__(self, denoise_fn, image_size, channels=3, conditional=True,
                 schedule_opt=None, opt=None, **kwargs):
        super().__init__()
        self.opt = opt
        self.cfg = opt['model']['uard']
        self.mode = self.cfg.get('mode', 'full')
        if self.mode not in self.MODES or not conditional:
            raise ValueError('UARD requires conditional input and a supported mode')
        self.residual_mode = self.mode in ('full', 'residual_uniform')
        self.channels, self.image_size = channels, image_size
        self.denoise_fn = denoise_fn
        self.w_min = float(self.cfg.get('w_min', 0.1))
        self.beta_T = float(self.cfg.get('beta_T', 1.0))
        lower = float(self.cfg.get('log_scale_min', -5.0))
        upper = float(self.cfg.get('log_scale_max', 1.0))
        if not 0 < self.w_min <= 1 or not math.isfinite(self.beta_T) or self.beta_T <= 0 or not lower < upper:
            raise ValueError('Require 0 < w_min <= 1, beta_T > 0 and log_scale_min < log_scale_max')
        self.anchor_predictor = OpticalAnchor(
            self.cfg.get('sar_channels', 3), channels, self.cfg.get('anchor_width', 32), lower, upper)
        if self.cfg.get('freeze_anchor', False):
            self.anchor_predictor.requires_grad_(False)
        self.loss_weights = {k: float(self.cfg.get(k, 1.0)) for k in
                             ('lambda_u', 'lambda_R', 'lambda_eps', 'lambda_rec')}
        if any(not math.isfinite(v) or v < 0 for v in self.loss_weights.values()):
            raise ValueError('Loss weights must be finite and nonnegative')
        self.conditioning = self.cfg.get('conditioning', {
            'full': 'sar_anchor_uncertainty', 'residual_uniform': 'sar_anchor_uncertainty',
            'anchor_condition': 'sar_anchor', 'uncertainty_only': 'sar_uncertainty'}[self.mode])
        if self.conditioning not in ('sar', 'sar_anchor', 'sar_uncertainty', 'sar_anchor_uncertainty'):
            raise ValueError('Unsupported denoiser conditioning')
        self.noise_loss = self.cfg.get('noise_loss', 'standard')
        if self.noise_loss not in ('standard', 'weighted'):
            raise ValueError('noise_loss must be standard or weighted')
        self.register_buffer('anchor_pretrained', torch.tensor(False))
        self.last_maps = {}
        train_schedule = opt['model']['beta_schedule']['train']
        self.schedule_signature = {k: train_schedule[k] for k in ('schedule', 'n_timestep', 'linear_start', 'linear_end')}

    def get_extra_state(self):
        return {'version': 2, 'kind': 'diffusion', 'conditioning': self.conditioning,
                'noise_loss': self.noise_loss, 'mode': self.mode, 'w_min': self.w_min, 'beta_T': self.beta_T,
                'log_scale_min': self.anchor_predictor.lower,
                'log_scale_max': self.anchor_predictor.upper, 'train_schedule': self.schedule_signature}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Checkpoint formulation differs from configuration; use matching UARD settings')

    def anchor_signature(self):
        return {'sar_channels': self.cfg.get('sar_channels', 3), 'optical_channels': self.channels,
                'anchor_width': self.cfg.get('anchor_width', 32),
                'log_scale_min': self.anchor_predictor.lower, 'log_scale_max': self.anchor_predictor.upper}

    def load_anchor_checkpoint(self, path):
        # Trusted local Stage-0 output only: old torch restricted loaders reject float metadata.
        state = torch.load(path, map_location='cpu', weights_only=False)
        expected = {'version': 2, 'kind': 'anchor_pretraining', 'predictor': self.anchor_signature()}
        if state.get('_extra_state') != expected:
            raise ValueError('Expected a matching Stage-0 anchor checkpoint, not a diffusion/legacy checkpoint')
        prefix = 'anchor_predictor.'
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.anchor_predictor.load_state_dict(weights, strict=True)
        self.anchor_pretrained.fill_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.get('freeze_anchor', False):
            self.anchor_predictor.eval()
        return self

    def make_condition(self, sar, anchor, uncertainty):
        pieces = [sar]
        if 'anchor' in self.conditioning:
            pieces.append(anchor)
        if 'uncertainty' in self.conditioning:
            if uncertainty is None:
                raise ValueError('Explicit uncertainty is required by this denoiser')
            pieces.append(uncertainty)
        return torch.cat(pieces, dim=1)

    def set_loss(self, device):
        pass  # Loss reductions below match the original trainer's sum / numel contract.

    def set_new_noise_schedule(self, schedule_opt, device, num_train_timesteps=1000):
        self.num_timesteps = int(num_train_timesteps)
        self.sampling_steps = int(schedule_opt['n_timestep']) if schedule_opt['ddim'] else self.num_timesteps
        if self.num_timesteps < 1:
            raise ValueError('num_train_timesteps must be positive')
        # VP baselines use E3Diff's schedule (including its approximate Gaussian terminal).
        from .diffusion import make_beta_schedule
        betas = torch.as_tensor(make_beta_schedule(
            schedule_opt['schedule'], self.num_timesteps,
            schedule_opt['linear_start'], schedule_opt['linear_end']), dtype=torch.float32, device=device)
        cumulative = torch.cat([torch.ones(1, device=device), (1 - betas).cumprod(0)])
        self.register_buffer('vp_signal', cumulative.sqrt(), persistent=False)
        self.register_buffer('vp_noise', (1 - cumulative).sqrt(), persistent=False)

    def coefficients(self, t, reference):
        t = torch.as_tensor(t, device=reference.device, dtype=torch.long).reshape(-1)
        # Sampling passes one shared timestep; E3Diff's time embedding needs
        # one entry per image, even though scalar coefficients broadcast in q_sample.
        if t.numel() == 1:
            t = t.expand(reference.shape[0])
        elif t.numel() != reference.shape[0]:
            raise ValueError('Time must be scalar or have one entry per batch item')
        if ((t < 0) | (t > self.num_timesteps)).any():
            raise ValueError('Time must lie in [0, T]')
        tau = (t.to(reference.dtype) / self.num_timesteps).view(-1, 1, 1, 1)
        if self.residual_mode:
            return tau, self.beta_T * tau.sqrt()
        return self.vp_signal[t].view(-1, 1, 1, 1), self.vp_noise[t].view(-1, 1, 1, 1)

    def predict_anchor(self, sar):
        if sar.ndim != 4 or sar.shape[1] != self.cfg.get('sar_channels', 3):
            raise ValueError('SAR must be BCHW with configured sar_channels')
        anchor, s, u = self.anchor_predictor(sar)
        weight = self.w_min + (1 - self.w_min) * u
        if self.mode in ('residual_uniform', 'anchor_condition'):
            weight = torch.ones_like(u)
        return anchor, s, u, weight

    def q_sample(self, target, anchor, weight, t, noise):
        raise NotImplementedError(
            "UARDiffusion.q_sample is not available in this source preview. "
            "See RELEASE_SCOPE.md for the release boundary."
        )

    def q_transition(self, previous_state, target, anchor, weight, previous_t, t, noise):
        """Sample forward Markov increment using independent standard noise.

        Its marginals match q_sample; its joint law differs from the shared-epsilon path.
        """
        if not self.residual_mode:
            raise ValueError('Residual Markov transition is only defined for C/E')
        a0, b0 = self.coefficients(previous_t, target)
        a1, b1 = self.coefficients(t, target)
        if (a1 <= a0).any():
            raise ValueError('Forward transition requires previous_t < t')
        return previous_state + (a1 - a0) * (anchor - target) + (b1.square() - b0.square()).clamp_min(0).sqrt() * weight * noise

    def predict(self, x, sar, anchor, weight, t, uncertainty=None):
        a, b = self.coefficients(t, x)
        condition = self.make_condition(sar, anchor, uncertainty)
        time = (1 - a.flatten(1)) if self.residual_mode else a.flatten(1)
        output = self.denoise_fn(torch.cat([condition, x], 1), time)
        if self.residual_mode:
            residual, epsilon = output.chunk(2, dim=1)
            clean = x - a * residual - b * weight * epsilon
        else:
            residual, epsilon = torch.zeros_like(x), output
            clean = (x - b * weight * epsilon) / a.clamp_min(1e-8)
        return clean, residual, epsilon

    def generate(self, sar, anchor, weight, steps, noise=None, seed=None, trajectory=False, uncertainty=None):
        raise NotImplementedError(
            "UARDiffusion.generate is not available in this source preview. "
            "See RELEASE_SCOPE.md for the release boundary."
        )

    def forward(self, data, noise=None):
        target, sar = data['HR'], data['SR']
        anchor, s, u, weight = self.predict_anchor(sar)
        if target.shape != anchor.shape:
            raise ValueError('Optical target and predicted anchor must have the same BCHW shape')
        # Shared spatial scale, mean over RGB: equivalent to per-channel Laplace NLL.
        unc = (torch.exp(-s) * (target - anchor).abs().mean(1, keepdim=True) + s).mean()
        # Train the predictive proxy by likelihood only, preventing diffusion shortcut gradients.
        fixed_anchor, fixed_weight, fixed_u = anchor.detach(), weight.detach(), u.detach()
        if noise is None:
            noise = torch.randn_like(target)
        losses = {'unc': unc, 'residual': target.new_zeros(()), 'noise': target.new_zeros(()),
                  'reconstruction': target.new_zeros(())}
        if self.opt['stage'] == 2:
            if self.opt.get('ddim_steps') != 1:
                raise ValueError('UARD Stage 2 requires ddim_steps=1')
            t = torch.full((target.shape[0],), self.num_timesteps, device=target.device, dtype=torch.long)
            pred, _ = self.generate(sar, fixed_anchor, fixed_weight, 1, noise=noise, uncertainty=fixed_u)
            losses['reconstruction'] = F.l1_loss(pred, target)
        else:
            t = torch.randint(1, self.num_timesteps + 1, (target.shape[0],), device=target.device)
            noisy = self.q_sample(target, fixed_anchor, fixed_weight, t, noise)
            pred, residual, epsilon = self.predict(noisy, sar, fixed_anchor, fixed_weight, t, uncertainty=fixed_u)
            noise_error = epsilon - noise
            if self.noise_loss == 'weighted':
                noise_error = fixed_weight * noise_error
            losses['noise'] = noise_error.square().mean()
            if self.residual_mode:
                losses['residual'] = F.l1_loss(residual, fixed_anchor - target)
        total = ((0.0 if self.cfg.get('freeze_anchor', False) else self.loss_weights['lambda_u']) * losses['unc']
                 + self.loss_weights['lambda_R'] * losses['residual']
                 + self.loss_weights['lambda_eps'] * losses['noise']
                 + self.loss_weights['lambda_rec'] * losses['reconstruction'])
        # A dict of scalar tensors is gathered safely by DataParallel.
        stats = {k: v.detach().reshape(1) * target.numel() for k, v in losses.items()}
        return total * target.numel(), target, pred, stats, t

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, seed=1, img_s1=None):
        anchor, s, u, weight = self.predict_anchor(x_in)
        self.last_maps = {'anchor': anchor, 'uncertainty': u, 'noise_weight': weight, 'log_scale': s}
        steps = 1 if self.opt['stage'] == 2 else self.sampling_steps
        result, history = self.generate(x_in, anchor, weight, steps, seed=seed, trajectory=continous, uncertainty=u)
        if continous:
            return history[0].clamp(-1, 1), history[1].clamp(-1, 1)
        return result.clamp(-1, 1), result.clamp(-1, 1)

    def sample(self, *args, **kwargs):
        raise ValueError('UARD inference requires SAR; call super_resolution(sar)')


class AnchorPretraining(UARDiffusion):
    """Stage 0 instantiates only G_phi; no diffusion backbone or auxiliary loss."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.opt['stage'] != 0 or self.cfg.get('freeze_anchor', False):
            raise ValueError('Stage 0 requires stage=0 and freeze_anchor=false')
        self.anchor_predictor.requires_grad_(True)

    def get_extra_state(self):
        return {'version': 2, 'kind': 'anchor_pretraining', 'predictor': self.anchor_signature()}

    def forward(self, data, noise=None):
        target, sar = data['HR'], data['SR']
        anchor, s, _, _ = self.predict_anchor(sar)
        if target.shape != anchor.shape:
            raise ValueError('Stage-0 target and anchor shapes differ')
        nll = (torch.exp(-s) * (target-anchor).abs().mean(1, keepdim=True) + s).mean()
        stats = {'unc': nll.detach().reshape(1)*target.numel()}
        return nll * target.numel(), target, anchor, stats, target.new_zeros(target.shape[0])

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, seed=1, img_s1=None):
        anchor, s, u, weight = self.predict_anchor(x_in)
        self.last_maps = {'anchor': anchor, 'uncertainty': u, 'noise_weight': weight, 'log_scale': s}
        return anchor, anchor
