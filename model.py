import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

# define the CNN architecture
class MyNet(nn.Module):
    ### choose an architecture, and complete the class
    def __init__(self, output_size, layer_sizes=[1024]):
        super(MyNet, self).__init__()
        ## Define layers of a CNN
        
        resnet = models.resnet101(pretrained=False) #.features
        #resnet = models.resnet18(pretrained=True) #.features
        
        # freeze all VGG parameters since we're only optimizing the target image
        for param in resnet.parameters():
            param.requires_grad_(False)
        # 512 * 7 * 7
        self.feat_dim  = resnet.fc.in_features

        #layers = [nn.Linear(in_features=self.feat_dim, out_features=1024), 
        #          nn.ReLU(inplace=True), 
        #          nn.Dropout(p=0.5),
        #          nn.Linear(in_features=1024, out_features=output_size)
        #         ]
        
        layers = []
        prev_ls = self.feat_dim
        for ls in layer_sizes:
            layers += [nn.Linear(in_features=prev_ls, out_features=ls), 
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.5)
                      ]
            prev_ls = ls
        layers += [nn.Linear(in_features=prev_ls, out_features=output_size)]
        
        resnet.fc = nn.Sequential(*layers)
        self.classifier = resnet
    
    def forward(self, x):
        ## Define forward behavior
        x = self.classifier(x) # extract features to first hidden dimension
        x = F.log_softmax(x, dim=1)
        return x

    def set_requires_grad(self, target_layers=[]):
        network = self.classifier
        # unfreeze the last resnet layers
        for cid, child in enumerate(network.children()):
            if cid in target_layers:
                for param in child.parameters():
                    param.requires_grad_(True)
            else:
                for param in child.parameters():
                    param.requires_grad_(False)
