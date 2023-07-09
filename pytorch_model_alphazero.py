import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    This class represents a Convolutional Block used as the first layer in AlphaZeroNet.
    It performs a 2D convolution followed by batch normalization and a ReLU activation.
    """

    def __init__(self, inplanes=7, planes=128):
        super(ConvBlock, self).__init__()
        # Define the convolutional layer
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        # Define the batch normalization layer
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        # Pass the input through the conv -> batchnorm -> ReLU pipeline
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    This class represents a Residual Block, which is the main component of the AlphaZeroNet.
    It performs two sets of 2D convolution, batch normalization, and ReLU activation with a skip connection.
    """

    def __init__(self, inplanes=128, planes=128):
        super(ResBlock, self).__init__()
        # Define the first conv -> batchnorm pipeline
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Define the second conv -> batchnorm pipeline
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """
    This class represents the Policy Head of the AlphaZeroNet.
    It predicts the probability distribution over all possible moves.
    """

    def __init__(self):
        """Initialize the PolicyHead."""
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(128, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 10 * 10, 978)

    def forward(self, x):
        """Perform a forward pass through the PolicyHead."""
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 2 * 10 * 10)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """
    This class represents the Value Head of the AlphaZeroNet.
    It predicts the value of the current board state.
    """

    def __init__(self):
        """Initialize the ValueHead."""
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(128, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """Perform a forward pass through the ValueHead."""
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 1 * 10 * 10)  # flatten
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class AlphaZeroNet(nn.Module):
    """
     This class represents the AlphaZeroNet which is a combination of ConvBlock, multiple ResBlocks, and PolicyHead and ValueHead.
     It returns the move probabilities and board state value for a given board state.
     """

    def __init__(self, name="unknown", device=torch.device("cpu")):
        """Initialize the AlphaZeroNet."""
        super(AlphaZeroNet, self).__init__()
        self.name = name
        self.device = device
        self.convblock = ConvBlock()
        self.res_blocks = nn.Sequential(*(ResBlock() for _ in range(10)))
        self.policyhead = PolicyHead()
        self.valuehead = ValueHead()
        self.to(self.device)

    def forward(self, x):
        """Perform a forward pass through the AlphaZeroNet."""
        x = self.convblock(x)
        x = self.res_blocks(x)
        return self.policyhead(x), self.valuehead(x)
