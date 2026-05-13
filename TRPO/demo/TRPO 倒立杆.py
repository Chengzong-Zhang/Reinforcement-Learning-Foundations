import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import gymnasium as gym


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = torch.as_tensor(action_bound, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_bound = self.action_bound.to(x.device)
        mu = action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy().flatten()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float32).view(-1, 1)


class TRPOContinuous:
    def __init__(
        self,
        hidden_dim,
        state_space,
        action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device,
    ):
        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)
        action_bound = np.maximum(np.abs(self.action_low), np.abs(self.action_high))
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        mu, std = self.actor(state.unsqueeze(0))
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample().detach().cpu().numpy()[0]
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def hessian_matrix_vector_product(
        self, states, old_action_dists, vector, damping=0.1
    ):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists).sum(
                dim=1, keepdim=True
            )
        )
        actor_params = tuple(self.actor.parameters())
        kl_grad = torch.autograd.grad(
            kl, actor_params, create_graph=True, allow_unused=False
        )
        kl_grad_vector = torch.cat([grad.contiguous().view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, actor_params)
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists, max_iterations=10):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for _ in range(max_iterations):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp).clamp_min(1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
        self, states, actions, advantage, old_log_probs, old_action_dists, max_vec
    ):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists, new_action_dists
                ).sum(dim=1, keepdim=True)
            )
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor
            )
            if new_obj.item() > old_obj.item() and kl_div.item() < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        actor_params = tuple(self.actor.parameters())
        grads = torch.autograd.grad(surrogate_obj, actor_params)
        obj_grad = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, direction)
        max_coef = torch.sqrt(
            2 * self.kl_constraint / torch.dot(direction, Hd).clamp_min(1e-8)
        )
        new_para = self.line_search(
            states,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            direction * max_coef,
        )
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )

    def update(self, transition_dict):
        states = torch.as_tensor(
            np.asarray(transition_dict["states"], dtype=np.float32),
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(transition_dict["actions"], dtype=np.float32),
            device=self.device,
        )
        rewards = torch.as_tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.as_tensor(
            np.asarray(transition_dict["next_states"], dtype=np.float32),
            device=self.device,
        )
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        rewards = (rewards + 8.0) / 8.0
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )

        with torch.no_grad():
            mu, std = self.actor(states)
            old_log_probs = (
                torch.distributions.Normal(mu, std)
                .log_prob(actions)
                .sum(dim=1, keepdim=True)
            )
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())

        critic_loss = F.mse_loss(self.critic(states), td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


def reset_env(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info
    next_state, reward, done, info = result
    return next_state, reward, done, info


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    episodes_per_iteration = max(1, num_episodes // 10)
    num_iterations = int(np.ceil(num_episodes / episodes_per_iteration))

    for iteration in range(num_iterations):
        start_episode = iteration * episodes_per_iteration
        end_episode = min(num_episodes, start_episode + episodes_per_iteration)
        with tqdm(total=end_episode - start_episode, desc=f"Iteration {iteration}") as pbar:
            for local_episode in range(end_episode - start_episode):
                episode_return = 0.0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }

                state = reset_env(env)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = step_env(env, action)
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (local_episode + 1) % 10 == 0 or local_episode + 1 == end_episode - start_episode:
                    pbar.set_postfix(
                        {
                            "episode": start_episode + local_episode + 1,
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )
                pbar.update(1)
    return return_list


def moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window_size:
        return values
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    radius = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[: window_size - 1])[::2] / radius
    end = (np.cumsum(values[: -window_size : -1])[::2] / radius)[::-1]
    return np.concatenate((begin, middle, end))


def plot_returns(return_list, env_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(len(return_list)))

    raw_path = output_dir / "trpo_pendulum_returns.png"
    smooth_path = output_dir / "trpo_pendulum_returns_smoothed.png"

    plt.figure()
    plt.plot(episodes, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"TRPO on {env_name}")
    plt.tight_layout()
    plt.savefig(raw_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(episodes, moving_average(return_list, 9))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"TRPO on {env_name}")
    plt.tight_layout()
    plt.savefig(smooth_path, dpi=150)
    plt.close()

    return raw_path, smooth_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRPO on Pendulum.")
    parser.add_argument("--env-name", type=str, default="Pendulum-v1")
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--critic-lr", type=float, default=1e-2)
    parser.add_argument("--kl-constraint", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(args.env_name)
    reset_env(env, seed=args.seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    agent = TRPOContinuous(
        args.hidden_dim,
        env.observation_space,
        env.action_space,
        args.lmbda,
        args.kl_constraint,
        args.alpha,
        args.critic_lr,
        args.gamma,
        device,
    )
    return_list = train_on_policy_agent(env, agent, args.num_episodes)
    env.close()

    raw_path, smooth_path = plot_returns(return_list, args.env_name, args.output_dir)
    print(f"Training finished on {device}.")
    print(f"Last 10 episode mean return: {np.mean(return_list[-10:]):.3f}")
    print(f"Saved plots: {raw_path}, {smooth_path}")


if __name__ == "__main__":
    main()
