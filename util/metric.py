import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score
import torch.nn.functional as F
from util.temperature_scaling import ModelWithTemperature
from util.bin_temperature_scaling import ModelWithBinwiseTemperature


def validation_accuracy_lora(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1) 
            outputs = model.forward_features(inputs)
            outputs = model.linear(outputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    # print("\n🔹 Accuracy Metrics 🔹")
    return valid_accuracy


def validation_accuracy(model, loader, device, mode='rein'):
    total = 0
    correct = 0

    # Output functions for different modes
    def linear(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs
    
    def full(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs
    
    def rein(model, inputs):
        if isinstance(model, ModelWithTemperature):
            f = model.model.forward_features(inputs) 
            f = f[:, 0, :]
            outputs = model.model.linear(f) 
        else:
            f = model.forward_features(inputs)  
            f = f[:, 0, :]  
            outputs = model.linear(f)  
        
        return outputs
    
    def adaptformer(model, inputs):
        f = model.forward_features(inputs)
        outputs = model.linear(f)
        return outputs


    def rein_dropout(model, inputs, num_samples=10):
        model.train() 
        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                f = model.forward_features(inputs)
                f = f[:, 0, :]
                output = model.linear(f)
                outputs.append(torch.softmax(output, dim=1))

        output = torch.mean(torch.stack(outputs), dim=0)
        return output

    def no_rein(model, inputs):
        f = model.forward_features_no_rein(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs
    
    def resnet(model, inputs):
        outputs = model(inputs)
        return outputs
    
    def densenet(model, inputs):
        outputs = model(inputs)
        return outputs
    
    def vgg(model, inputs):
        outputs = model(inputs)
        return outputs

    # Select appropriate output function
    if mode == 'rein':
        out = rein
    elif mode == 'no_rein':
        out = no_rein
    elif mode == 'adaptformer':
        out = adaptformer
    elif mode == 'linear':
        out = linear
    elif mode == 'full':
        out = full

    if mode == 'rein_dropout':
        model.train()
    else:
        model.eval()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1)
            
            outputs = out(model, inputs)
            _, predicted = outputs.max(1)

            # Ensure the same batch size for predicted and targets
            min_batch_size = min(predicted.size(0), targets.size(0))
            predicted = predicted[:min_batch_size]
            targets = targets[:min_batch_size]

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    valid_accuracy = correct / total
    return valid_accuracy



