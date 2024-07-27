import torch.nn as nn
import torch
import os
from typing import Union

"""
Actor: Returns predicted action with some probability 
"""
class Actor(nn.Module):
    def __init__(self, pretrained_model) -> None:
        """
        pretrained_model: Trained imitation learning model used to initialize the Actor
        """
        super(Actor, self).__init__()

        # initialize Actor with layers and weights of the pre-trained imitation learning model
        self.conv_blocks = pretrained_model.conv_blocks
        self.img_fc = pretrained_model.img_fc
        self.scalar_fc = pretrained_model.scalar_fc
        self.emb_fc = pretrained_model.emb_fc
        self.branches = pretrained_model.branches

    def forward(self, x_img, scalar, command):
        x_img = self.conv_blocks(x_img)
        x_img = self.img_fc(x_img)

        scalar = scalar[:, 0].unsqueeze(1)
        scalar = self.scalar_fc(scalar)

        emb = torch.cat([x_img, scalar], dim=1)
        emb = self.emb_fc(emb)

        # use the appropriate branch based on the command
        output_list = []
        for i in range(emb.shape[0]):  # Iterate over batch size
            branch_output = self.branches[command[i]](emb[i])
            output_list.append(branch_output.unsqueeze(0))

        action = torch.cat(output_list, dim=0)
        return action


"""
Critic: Estimates the Q-value which determines the correctness of the action outputted by the Actor
"""
class Critic(nn.Module):
    def __init__(self, action_dim, pretrained_model) -> None:
        """
        pretrained_model: Trained imitation learning model used to initialize the Actor
        action_dim: Number of actions, default is 3 (steer, throttle, brake)
        """
        super(Critic, self).__init__()

        self.conv_blocks = pretrained_model.conv_blocks
        self.img_fc = pretrained_model.img_fc
        self.scalar_fc = pretrained_model.scalar_fc

        # adjust embedding layers for critic to include action dimension (3) as input
        self.emb_fc = nn.Sequential(
            nn.Linear(512 + 128 + action_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # adjust final FC layer to only output 1 value (Q-val) instead of 3 actions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(3)
        ])

    def forward(self, x_img, scalar, command, action):
        x_img = self.conv_blocks(x_img)
        x_img = self.img_fc(x_img)

        scalar = scalar[:, 0].unsqueeze(1)
        scalar = self.scalar_fc(scalar)

        emb = torch.cat([x_img, scalar, action], dim=1)
        emb = self.emb_fc(emb)

        # use the appropriate branch based on the command
        output_list = []
        for i in range(emb.shape[0]):  # Iterate over batch size
            branch_output = self.branches[command[i]](emb[i])
            output_list.append(branch_output.unsqueeze(0))

        q_value = torch.cat(output_list, dim=0)
        return q_value


class IL_Model(nn.Module):
    def __init__(self):
        super(IL_Model, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.scalar_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.emb_fc = nn.Sequential(
            nn.Linear(512+128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for _ in range(3)
        ])

    def forward(self, x_img, scalar, command):
        x_img = self.conv_blocks(x_img)
        x_img = self.img_fc(x_img)

        scalar = scalar[:, 0].unsqueeze(1)
        scalar = self.scalar_fc(scalar)

        emb = torch.cat([x_img, scalar], dim=1)
        emb = self.emb_fc(emb)

        output_list = []
        for i in range(emb.shape[0]):  # Iterate over batch size
            branch_output = self.branches[command[i]](emb[i])
            output_list.append(branch_output.unsqueeze(0))

        output = torch.cat(output_list, dim=0)
        return output


def load_model(model_path, device, base_model: Union[IL_Model, Actor, Critic, torch.optim.Adam]):
    """
    Loads a saved model.

    Parameters:
    - model_path (str): path to the saved model
    - device (torch device): device to store model on
    - base_model (model): A newly initialized model (IL_model, Actor, Critic, or PyTorch Adam)
    """
    if model_path is None:
        raise AttributeError("The model path is None.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model path does not exist.")
    
    try:
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Imitation Learning model loaded")
    except FileNotFoundError:
        print("Imitation Learning model not found")
        return
    
    return base_model

def save_model(folder, model, name):
    """
    Saves the model to the specified folder with the specified name.

    Parameters:
        folder (str): The folder in which to save the model.
        model (torch.nn.Module): The model to save.
        name (str): The name to use for the saved model file.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.save(model.state_dict(), os.path.join(folder, f"{name}.pth"))
    print(f"Model saved to {folder}/{name}.pth")