"""
Custom rsl_rl ActorCritic for the camera variant: Nature CNN encoder -> [64,64] MLP
heads (plan.md Phase 3's own words: "no SO-101 reference exists for this... per 'pick
what you think is best': use the standard Nature CNN encoder (3 conv layers, channels
32->64->64, strides 4/2/1, ReLU activations, flatten) ahead of the same [64, 64]
actor/critic MLP heads").

UPDATE (2026-08-10): CONFIRMED BROKEN, not just a hypothetical forward-compat risk
anymore. Directly pip-installed and inspected rsl-rl-lib: `rsl_rl.modules.ActorCritic`
does not exist in the latest release (5.4.2) OR in 5.0.1 specifically -- the exact
version Isaac Lab's `main` branch pins in `source/isaaclab_rl/setup.py` (confirmed by
reading that file directly, in the research this docstring already describes below).
Both were checked, not assumed. `rsl_rl.modules` in both only exposes the new modular
split (`CNN`, `MLP`, `GaussianDistribution`, etc.) -- this custom class's entire "OLD
single-class API is the safer target" premise does not hold for whatever Phase 1
actually installs, if it matches what Isaac Lab's own setup.py already specifies. The
`RslRlCNNModelCfg`/`RslRlMLPModelCfg` path described below is the one to use FIRST, not
a fallback to check only if this file breaks -- this file is now the fallback-of-a-
fallback. `scripts/export_policy.py`'s lidar export path already fails loudly with an
actionable message if it hits the same missing import, rather than a bare ImportError.
  `RslRlCNNModelCfg(cnn_cfg=CNNCfg(output_channels=[32,64,64], kernel_size=[8,4,3],
  stride=[4,2,1], activation="relu", flatten=True), hidden_dims=[64,64],
  activation="elu", distribution_cfg=GaussianDistributionCfg(init_std=1.0))`
  passed directly as `RslRlOnPolicyRunnerCfg`'s `actor=`/`critic=` fields -- this maps
  onto plan.md's CNN spec almost verbatim, natively, no custom class needed.
**Check for `isaaclab_rl.rsl_rl.RslRlCNNModelCfg` first in Phase 1** (`python -c "from
isaaclab_rl.rsl_rl import RslRlCNNModelCfg"`) -- if it's importable, use it instead of
this file; it's the real, maintained path, not a guess. Two things are NOT confirmed
even for that newer path (docs don't show them, and the raw `CNN` module in rsl_rl only
takes a single tensor, so the multi-group routing lives in a higher-level wrapper this
research didn't turn up): (a) exactly how `obs_groups={"actor": ["policy", "images"],
...}` routes an "images" group through the CNN vs. a "policy" (proprioceptive) group
around it into the MLP head, and (b) the exact `ObservationsCfg` shape Isaac Lab expects
for that split (this repo's env_cfg_camera.py would need restructuring into two named
observation groups, not the current single un-concatenated "policy" group). Confirm
both against a real install before switching.

If neither path works as researched, this custom class (below) is the fallback -- its
own base-class contract (method names, what `act`/`evaluate` must return, how
`.distribution`/`.action_mean`/`.action_std`/`.entropy` behave) is reconstructed from
the OLD `rsl_rl.modules.actor_critic.ActorCritic` shape and was not checked against a
running install. Compare against whatever's actually installed before trusting it
verbatim. Separately: env_cfg_camera.py sets `concatenate_terms = False` so the image
arrives unflattened -- if `RslRlVecEnvWrapper` in your installed version insists on a
single flat "policy" tensor instead of a dict, the fallback is to flatten the image into
that same vector and reshape it back to (C,H,W) inside this class's `_split_obs`
instead -- uglier, but only requires `concatenate_terms=True` and no wrapper changes.

If this integration doesn't pan out as written, that's expected -- it's flagged here
precisely so Phase 2 checks it deliberately instead of discovering it by surprise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _nature_cnn(in_channels: int = 3) -> nn.Sequential:
    """3 conv layers, channels 32/64/64, strides 4/2/1, ReLU, flatten -- the standard
    DQN-paper encoder, per plan.md's own citation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


def _cnn_output_dim(cnn: nn.Sequential, image_shape: tuple[int, int, int]) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, *image_shape)
        return cnn(dummy).shape[-1]


class NatureCnnActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_proprio_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (3, 480, 640),  # (C, H, W) -- matches CameraCfg's 640x480 (env_cfg_camera.py)
        # Tuples, not lists -- a mutable list default here is a classic Python footgun
        # (shared across every instance constructed without passing this arg
        # explicitly; harmless today since make_head() below only ever iterates
        # these, never mutates them in place, but a real risk if that ever changes).
        # Found by static analysis, not by reading the constructor body.
        actor_hidden_dims: tuple[int, ...] = (64, 64),
        critic_hidden_dims: tuple[int, ...] = (64, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        act_cls = {"elu": nn.ELU, "relu": nn.ReLU}[activation]

        self.actor_cnn = _nature_cnn(image_shape[0])
        self.critic_cnn = _nature_cnn(image_shape[0])  # separate encoder, not shared -- simpler and avoids actor-gradient-through-critic-features surprises; revisit if VRAM is tight (Phase 2)
        cnn_out_dim = _cnn_output_dim(self.actor_cnn, image_shape)

        def make_head(hidden_dims: list[int], out_dim: int) -> nn.Sequential:
            layers = []
            in_dim = cnn_out_dim + num_proprio_obs
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), act_cls()]
                in_dim = h
            layers += [nn.Linear(in_dim, out_dim)]
            return nn.Sequential(*layers)

        self.actor_head = make_head(actor_hidden_dims, num_actions)
        self.critic_head = make_head(critic_hidden_dims, 1)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _split_obs(observations, image_shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        observations arrives as whatever env_cfg_camera.py's un-concatenated policy
        obs group + the vec-env wrapper produce -- written expecting a dict with
        "image" and the proprioceptive terms, falling back to a manual split of a flat
        tensor if the wrapper ends up concatenating after all (see module docstring
        point 2). Whichever path is wrong for your actual installed wrapper, fix here,
        not by threading a shape assumption through the rest of this class.
        """
        if isinstance(observations, dict):
            image = observations["image"]
            proprio = torch.cat([observations["base_pose"], observations["base_velocity"]], dim=-1)
        else:
            c, h, w = image_shape
            image_flat_dim = c * h * w
            image = observations[:, :image_flat_dim].reshape(-1, c, h, w)
            proprio = observations[:, image_flat_dim:]
        if image.shape[-1] in (1, 3, 4) and image.shape[1] not in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W) if it arrived channel-last
        return image.float() / 255.0, proprio.float()

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        image, proprio = self._split_obs(observations, (3, 480, 640))
        features = self.actor_cnn(image)
        mean = self.actor_head(torch.cat([features, proprio], dim=-1))
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        image, proprio = self._split_obs(observations, (3, 480, 640))
        features = self.actor_cnn(image)
        return self.actor_head(torch.cat([features, proprio], dim=-1))

    def evaluate(self, critic_observations, **kwargs):
        image, proprio = self._split_obs(critic_observations, (3, 480, 640))
        features = self.critic_cnn(image)
        return self.critic_head(torch.cat([features, proprio], dim=-1))
