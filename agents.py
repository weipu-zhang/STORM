import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import copy
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar
from sub_models.constants import DEVICE, DTYPE_16


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    if DEVICE.type == "mps":
        # MPS does not support kthvalue, use sorting instead
        sorted_x, _ = torch.sort(flat_x)
        per = sorted_x[kth]
    else:
        per = torch.kthvalue(flat_x, kth + 1).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device=device)
    gamma_return = torch.zeros(
        (batch_size, batch_length + 1), dtype=dtype, device=DEVICE.type
    )
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(
        self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        # Sequential actor model to map from feat_dim to action_dim
        actor_model = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            actor_model.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.actor = nn.Sequential(*actor_model, nn.Linear(hidden_dim, action_dim))

        # Sequential critic model to map from feat_dim to 255 dim
        critic_model = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            critic_model.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.critic = nn.Sequential(*critic_model, nn.Linear(hidden_dim, 255))
        # Make a copy of critic for slow critic
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        # Enable scaler based on DEVICE type
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if DEVICE.type == "cuda"
            else None
        )

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        """
        Update slow critic models parameters with decay.
        """
        for slow_param, param in zip(
            self.slow_critic.parameters(), self.critic.parameters()
        ):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        """
        Use the logits of the actor model to get the policy distribution.
        """
        logits = self.actor(x)
        return logits

    def value(self, x):
        """
        Use the critic model to get the value of the state.
        """
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        """
        Use the slow critic model to get the slow-value of the state.
        """
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        """
        Get the raw actor logits and raw critiic value from the actor and critic models.
        """
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        """
        Get the action using the policy distribution (Actor model) from the latent state.
        Based on greedy or sampling.
        """
        self.eval()
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):

            action_logits = self.policy(latent)
            action_dist = distributions.Categorical(logits=action_logits)
            if greedy:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()

    def update(
        self, latent, action, old_logprob, old_value, reward, termination, logger=None
    ):
        """
        Update policy and value model
        """
        self.train()
        with torch.autocast(
            device_type=DEVICE.type, dtype=DTYPE_16, enabled=self.use_amp
        ):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(
                reward, slow_value, termination, self.gamma, self.lambd
            )
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(
                reward, value, termination, self.gamma, self.lambd
            )

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], lambda_return.detach()
            )
            slow_value_regularization_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], slow_lambda_return.detach()
            )

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound - lower_bound
            norm_ratio = torch.max(
                torch.ones(1).to(DEVICE), S
            )  # max(1, S) in the paper
            norm_advantage = (lambda_return - value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = (
                policy_loss
                + value_loss
                + slow_value_regularization_loss
                - self.entropy_coef * entropy_loss
            )

        # gradient descent
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log("ActorCritic/policy_loss", policy_loss.item())
            logger.log("ActorCritic/value_loss", value_loss.item())
            logger.log("ActorCritic/entropy_loss", entropy_loss.item())
            logger.log("ActorCritic/S", S.item())
            logger.log("ActorCritic/norm_ratio", norm_ratio.item())
            logger.log("ActorCritic/total_loss", loss.item())
