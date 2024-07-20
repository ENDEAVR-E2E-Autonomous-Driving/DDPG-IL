import torch.nn as nn
import torch

"""
Actor: Returns predicted action with some probability 
"""
class Actor(nn.Module):
    def __init__(self, pretrained_model) -> None:
        super(Actor, self).__init__()

        # initialize Actor with layers and weights of the pre-trained imitation learning model
        self.conv_blocks = pretrained_model.conv_blocks
        self.img_fc = pretrained_model.image_fc
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
        super(Critic, self).__init__()

        self.conv_blocks = pretrained_model.conv_blocks
        self.img_fc = pretrained_model.image_fc
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




    