import torch
import torch.nn as nn
import torch.optim as optim


# 门控网络
class Gating(nn.Module):
    """[b,4096*2]->[b,2]"""
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 4096)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(4096, 1024)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(1024, 512)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(512, 128)
        self.leaky_relu3 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.layer5 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.leaky_relu3(x)
        x = self.dropout4(x)

        return torch.softmax(self.layer5(x), dim=1)

class Class_projecyion(nn.Module):
    """[b,4096*2]->[b,2]"""
    def __init__(self, input_dim,
                 num_classes, dropout_rate=0.1):
        super(Class_projecyion, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(1024, 512)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(512, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)

class DoctorProjection(nn.Module):
    """[b,1024]->[b,4096]"""
    def __init__(self, input_dim,
                 output_dim, dropout_rate=0.1):
        super(DoctorProjection, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, input_dim*2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.leaky_relu1 = nn.LeakyReLU()

        self.layer2 = nn.Linear(input_dim*2, input_dim * 2)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(input_dim*2, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        return x

class MoE(nn.Module):
    def __init__(self, num_experts,x1dim,x2dim):
        super(MoE, self).__init__()
        # self.experts = nn.ModuleList(trained_experts)
        # num_experts = len(trained_experts)
        # # Assuming all experts have the same input dimension
        # input_dim = trained_experts[0].layer1.in_features
        input_dim = 4096*2
        self.gating = Gating(input_dim, num_experts)
        self.pdf_doctor = DoctorProjection(x2dim,x1dim)
        self.cls_doctor = Class_projecyion(4096,3)

    def forward(self, x1, x2):
        x2 = self.pdf_doctor(x2)
        x = torch.cat((x1,x2),dim=1)
        # Get the weights from the gating network
        weights = self.gating(x)
        outputs = weights[:,0] * x1 + weights[:,1] * x2
        cls = self.cls_doctor(outputs)
        return outputs, cls

# 数据集
# Generate the dataset
num_samples = 5000
input_dim = 4
hidden_dim = 32
x1 = torch.randn((1,4096))
x2 = torch.randn((1,5120))
models = MoE(2,x1dim=x1.shape[1],x2dim=x2.shape[1])

outputs, cls = models(x1,x2)
print(outputs.shape, cls.shape)


