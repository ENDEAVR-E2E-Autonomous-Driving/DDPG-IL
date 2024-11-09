import numpy as np

"""Ornstein-Uhlenbeck Noise (encourage exploration of actions)"""
class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.initial_std_dev = std_dev  # Store the initial standard deviation for reference
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros_like(mean)
        self.x_prev = np.copy(self.x0)

    def __call__(self, action):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x

        noisy_actions = action + x

        # ensure actions are within their correct range after adding noise
        noisy_actions[0] = np.clip(noisy_actions[0], -1, 1) # steer
        noisy_actions[1] = np.clip(noisy_actions[1], 0, 1) # throttle
        noisy_actions[1] = max(noisy_actions[1], 0.3)  # Ensure at least 20% throttle
        noisy_actions[2] = np.clip(noisy_actions[2], 0, 1) # brake

        return noisy_actions

    def reset(self):
        self.x_prev = self.x0

    def decay_noise(self, decay_factor=0.95, min_std_dev=0.05):
        """Decays the standard deviation of the noise, but not below min_std_dev to ensure exploration."""
        self.std_dev = np.maximum(min_std_dev, self.std_dev * decay_factor)
