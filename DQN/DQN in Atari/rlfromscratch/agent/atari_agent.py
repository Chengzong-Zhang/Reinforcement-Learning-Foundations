import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
from typing import Union

from utils.networks import AtariDQNNetwork
from utils.utils import atari_state_preprocess_function


def to_correct_device_tensor(input, device, dtype=torch.float32)-> torch.Tensor:
    # Converts input (numpy array or torch tensor) to a tensor on the target device
    if isinstance(input, np.ndarray):
        # from_numpy shares CPU memory (no copy); .to() fuses dtype conversion + GPU transfer in one pass
        # This avoids the intermediate CPU float32 copy that torch.tensor() always makes
        return torch.from_numpy(np.ascontiguousarray(input)).to(dtype=dtype, device=device)
    elif isinstance(input, torch.Tensor):
        return input.to(dtype=dtype, device=device)
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    def __init__(self,observation_space: Space, action_space: Space, device):
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space 
        self.device = device
        
    
    @abstractmethod
    def select_action(self, states:np.ndarray, deterministic:bool) -> np.ndarray:
        """Choose the action according to the states agent observed. """
        pass 
    
    def _get_other_parameters(self):  # prefix _ means: internal use only, not meant to be called from outside the class
        return dict(
            observation_space = self.observation_space,
            action_space = self.action_space,
            device = self.device 
        )

    def save(self, pth_file: str):
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_other_parameters()}, pth_file)
        
    def load(self,path:str):
        pth_file = torch.load(path,weights_only=False)
        self.load_state_dict(pth_file['state_dict'])
        # TODO: check when other parameters are added to Agent
        data = pth_file['data']
        # self.observation_space = data['observation_space']
        # self.action_space = data['action_space']
        # self.device = data['device']
        for key, value in data.items():
            setattr(self, key, value)  # setattr(obj, name, value) ≡ obj.name = value, but name is a runtime string
        


class AtariDQNAgent(AgentBase):
    def __init__(self, observation_space: Space, action_space: Space, device, alpha=1.0):
        super(AtariDQNAgent, self).__init__(observation_space, action_space, device)
        self.num_actions = action_space.n  # type: ignore[union-attr]
        self.observation_shape = observation_space.shape  # type: ignore[assignment]
        self.network = AtariDQNNetwork(observation_space.shape, action_space.n)  # type: ignore[union-attr, arg-type]
        self.alpha = alpha
        # CNN that maps (B,4,84,84) → (B, num_actions) Q-values
    

    def select_action(self, states:np.ndarray, deterministic=False) -> np.ndarray:
        # TODO: select actions
        state_t = to_correct_device_tensor(states, self.device)  # uint8 → GPU (one fused transfer)
        if isinstance(states, np.ndarray) and states.dtype == np.uint8:
            state_t = state_t / 255.0  # normalize on GPU: 4× less PCIe data vs CPU float32

        single = state_t.ndim == 3  # (C, H, W) → single state; (B, C, H, W) → batched
        if single:
            state_t = state_t.unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():  # no gradient needed during action selection
            q_values = self.network(state_t)  # (B, num_actions)

        if deterministic:
            # Greedy: pick the action with the highest Q-value
            actions = q_values.argmax(dim=1)  # (B,)
            return actions.item() if single else actions.cpu().numpy()
        else:
            # Softmax exploration: scale Q-values by temperature alpha, then sample
            scaled = q_values / self.alpha                             # divide by temperature α
            probs = torch.softmax(scaled, dim=1)                      # (B, num_actions)
            probs_np = probs.cpu().numpy()                            # move to CPU for numpy sampling
            if single:
                return np.random.choice(self.num_actions, p=probs_np[0])
            return np.array([np.random.choice(self.num_actions, p=p) for p in probs_np])
    

    def get_q(self, states:np.ndarray, actions:Union[np.ndarray, torch.Tensor]):
        """
        params: state: (batch, channel, 84,84)
        output: action: (batch, 1)
        """
        # TODO: compute Q(s,a)
        # The network already computes Q(s, a) for all actions at once.
        # What we need to do is get the full Q-value matrix and index out the value for each executed action.
        states_t   = to_correct_device_tensor(states, self.device)                    # uint8 → GPU (one fused transfer)
        if isinstance(states, np.ndarray) and states.dtype == np.uint8:
            states_t = states_t / 255.0                                               # normalize on GPU
        actions_t  = to_correct_device_tensor(actions,self.device, dtype=torch.int64) # gather requires int64
        q_values   = self.network(states_t)                                           # (B, num_actions)
        index      = actions_t.reshape(-1, 1)                                          # (B,) or (B,1) → always (B, 1)
        q_selected = torch.gather(q_values, dim=1, index=index)                      # (B, 1)
        return q_selected.squeeze(1)                                                  # (B, 1) → (B,)

    def get_max_q(self, states:np.ndarray):
        # TODO: compute max_a Q(s,a)
        states_t   = to_correct_device_tensor(states, self.device)                    # uint8 → GPU (one fused transfer)
        if isinstance(states, np.ndarray) and states.dtype == np.uint8:
            states_t = states_t / 255.0                                               # normalize on GPU
        with torch.no_grad():                                                          # no gradient needed for target Q
            q_values   = self.network(states_t)                                       # (B, num_actions)
        max_q      = q_values.max(dim=1).values                                       # (B,)
        return max_q
        
        


