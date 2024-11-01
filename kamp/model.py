import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import math
from tqdm import tqdm

class InceptionModule(nn.Module):
    def __init__(self, input_channels, num_feature_maps):
        super(InceptionModule, self).__init__()

        # brach 1
        self.branch_1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_feature_maps['branch_1'],
            kernel_size=1,
            stride=1
        )
        
        # branch 2
        self.branch_2_1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_feature_maps['branch_2_1'],
            kernel_size=1,
            stride=1
        )
        self.branch_2_2 = nn.Conv1d(
            in_channels=num_feature_maps['branch_2_1'],
            out_channels=num_feature_maps['branch_2_2'],
            kernel_size=3,
            padding=(3-1)//2
        )

        # brach 3
        self.branch_3_1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_feature_maps['branch_3_1'],
            kernel_size=1,
            stride=1
        )
        self.branch_3_2 = nn.Conv1d(
            in_channels=num_feature_maps['branch_3_1'],
            out_channels=num_feature_maps['branch_3_2'],
            kernel_size=5,
            stride=1,
            padding=(5-1)//2
        )

        # branch 4
        self.branch_4_1 = nn.AvgPool1d(
            kernel_size=3,
            stride=1,
            padding=(3-1)//2
        )
        self.branch_4_2 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_feature_maps['branch_4'],
            kernel_size=1,
            stride=1
        )
    
    def forward(self, x):
        branch_1 = self.branch_1(x)

        branch_2 = self.branch_2_1(x)
        branch_2 = self.branch_2_2(branch_2)

        branch_3 = self.branch_3_1(x)
        branch_3 = self.branch_3_2(branch_3)

        branch_4 = self.branch_4_1(x)
        branch_4 = self.branch_4_2(branch_4)

        concat = torch.cat((branch_1, branch_2, branch_3, branch_4), dim=1)

        return concat

class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()

        self.fc_1 = nn.Linear(in_features=17, out_features=10)
        self.bn_1 = nn.BatchNorm1d(10)
        self.relu_1 = nn.ReLU()

        self.fc_2 = nn.Linear(in_features=10, out_features=6)
        self.bn_2 = nn.BatchNorm1d(6)
        self.relu_2 = nn.ReLU()

        self.fc_3 = nn.Linear(in_features=6, out_features=2)
        self.softmax = nn.Softmax(dim=-1)

        init.kaiming_uniform_(self.fc_1.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.fc_2.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.fc_3.weight, a=math.sqrt(5))

        if self.fc_1.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc_1.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.fc_1.bias, -bound, bound)

        if self.fc_2.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc_2.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.fc_2.bias, -bound, bound)

        if self.fc_3.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc_3.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.fc_3.bias, -bound, bound)
    
    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.fc_3(x)
        x = self.softmax(x)

        return x


class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()

        # inception layer 1
        num_feature_maps_1 = {
            'branch_1' : 512,
            'branch_2_1' : 512,
            'branch_2_2' : 256,
            'branch_3_1' : 256,
            'branch_3_2' : 128,
            'branch_4': 512
        }
        self.inception_layer_1 = InceptionModule(
            input_channels = 1,
            num_feature_maps = num_feature_maps_1
        )

        # inception layer 2
        num_feature_maps_2 = {
            'branch_1' : 256,
            'branch_2_1' : 256,
            'branch_2_2' : 128,
            'branch_3_1' : 128,
            'branch_3_2' : 64,
            'branch_4': 256
        }
        self.inception_layer_2 = InceptionModule(
            input_channels = num_feature_maps_1['branch_1'] 
                                + num_feature_maps_1['branch_2_2'] 
                                + num_feature_maps_1['branch_3_2'] 
                                + num_feature_maps_1['branch_4'],
            num_feature_maps = num_feature_maps_2
        )

        # inception layer 3
        num_feature_maps_3 = {
            'branch_1' : 128,
            'branch_2_1' : 128,
            'branch_2_2' : 64,
            'branch_3_1' : 64,
            'branch_3_2' : 32,
            'branch_4': 128
        }
        self.inception_layer_3 = InceptionModule(
            input_channels = num_feature_maps_2['branch_1'] 
                                + num_feature_maps_2['branch_2_2'] 
                                + num_feature_maps_2['branch_3_2'] 
                                + num_feature_maps_2['branch_4'],
            num_feature_maps = num_feature_maps_3
        )

        # global avg pool
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # classifier
        self.classifier = ClassifierModule()

    
    def forward(self, x):
        x = self.inception_layer_1(x)
        x = self.inception_layer_2(x)
        x = self.inception_layer_3(x)
        x = self.global_avg_pool(x.permute(0, 2, 1))
        x = self.classifier(x.permute(0, 2, 1).squeeze(1))

        return x


class KampInceptionNet:
    def __init__(self, criterion=None, optimizer=None, lr=0.01, epochs=10):
        self.model = InceptionModel()
        self.lr = lr
        self.epochs = epochs
        self.history = []

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else :
            self.criterion = criterion
        
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        else:
            self.optimizer = optimizer
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
    def fit(self, dataloader):
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            iterator = tqdm(dataloader)
            for x_batch, y_batch in iterator:
                self.optimizer.zero_grad()

                x_batch = x_batch.unsqueeze(1)
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)

                loss.backward()

                self.optimizer.step()

                self.scheduler.step()

                epoch_loss += loss.item()
            
                iterator.set_description_str(f"[Epoch {epoch + 1} | Loss {epoch_loss/len(dataloader):.4f}] ")
            
            self.history.append(epoch_loss)
        
        return self.model, self.history