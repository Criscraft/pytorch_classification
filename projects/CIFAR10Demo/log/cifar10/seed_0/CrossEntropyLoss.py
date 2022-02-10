import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    def set_weights(self, weights):
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weights, reduction='sum')


    def forward(self, config, device, batch, model):
        #prepare
        batch_size = batch['data'].shape[0]

        #forward pass
        model_output = model(batch)
        
        #compute loss
        loss = self.cross_entropy_loss(model_output['logits'], batch['label'])

        #record statistics
        with torch.no_grad():
            pred = F.log_softmax(model_output['logits'], dim=1).argmax(1)
            correct = (pred == batch['label']).sum().float()
            #get batch size for statistics
            batch_size = torch.tensor(batch_size, dtype=torch.float, device=device, requires_grad=False)

        out_dict = {
            'loss' : loss,
            'correct' : correct,
            'batch_size' : batch_size,
        }
        return out_dict