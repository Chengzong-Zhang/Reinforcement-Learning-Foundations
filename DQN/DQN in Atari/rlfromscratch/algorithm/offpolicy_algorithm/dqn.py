from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent.atari_agent import AgentBase, AtariDQNAgent


OPTIMIZER_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}


class DQN(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent:AgentBase, logger: Logger, device, save_pth: str, args, target_agent=None):
        super(DQN, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, args)
        
        algo_args = args.algorithm
        assert algo_args.name=="DQN", "The method name in args must be 'dqn' for DQN algorithm."
        self.lr = algo_args.learning_rate
        self.gamma = algo_args.gamma 
        self.start_epsilon = algo_args.start_epsilon
        self.epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_timestep = algo_args.epsilon_timestep
        self.epsilon_schedular = algo_args.epsilon_schedular
        self.use_target = algo_args.use_target 
        self.batch_size = algo_args.batch_size
        self.device = device
        
        if self.use_target:
            self.target_agent = target_agent.to(self.device)
            self._target_hard_update()

        self.target_update_method = algo_args.target_update_method
        self.target_update_interval = algo_args.target_update_interval
        self.tau = algo_args.target_update_tau


        
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.parameters(), lr=self.lr)
        
        
    
    def _target_hard_update(self):
        # TODO: hard update 
        self.target_agent.load_state_dict(self.agent.state_dict())
    
    def _target_soft_update(self):
        # TODO: do soft update 
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _update_buffer(self, batch):
        self.buffer.add(batch)


    def _update_policy(self):
        # TODO: compute DQN loss according to paper 
        result = Result("policy")
        batch = self.buffer.sample(self.batch_size)

        q = self.agent.get_q(batch['states'],batch['actions'])

        with torch.no_grad():
            if self.use_target:
                max_next_q = self.target_agent.get_max_q(batch['next_states'])
            else:
                max_next_q = self.agent.get_max_q(batch['next_states'])
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        dones   = torch.tensor(batch['dones'],   dtype=torch.float32, device=self.device)

        td_target = rewards + self.gamma * max_next_q * (1.0 - dones)
        td_error = td_target - q
        loss = LOSS_DICT["huber"](q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target:
            if self.target_update_method == "hard":
                if self.gradient_step % self.target_update_interval == 0:
                    self._target_hard_update()
            else:  # soft
                self._target_soft_update()

        result.add_metric("network/loss", loss.item())
        result.add_metric("td_error", td_error.mean().item())
        result.add_metric("q_value_mean", q.mean().item())
       

        self.gradient_step += 1
        return result

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        # decay epsilon linearly from start_epsilon to end_epsilon over epsilon_timestep steps
        self.epsilon = max(
            self.end_epsilon,
            self.start_epsilon - (self.start_epsilon - self.end_epsilon) * self.interaction_step / self.epsilon_timestep
        )
        return np.random.rand() < self.epsilon

    def interact_with_envs(self):
        batch, result = super().interact_with_envs()
        result.add_metric("epsilon",self.epsilon)
        return batch, result