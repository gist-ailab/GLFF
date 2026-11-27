
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from calibrate.evaluation.metrics import ECELoss
from .temperature_scaling import _ECELoss


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    # output = torch.softmax(output, dim=1)

    return output


class ModelWithTemperature(nn.Module):
    """
    개별 모델에 대해 Temperature Scaling을 적용하는 클래스
    """
    def __init__(self, model, device='cuda:0'):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.device = device

    def forward(self, inputs):
        logits = rein_forward(self.model, inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature.to(logits.device)

    def set_temperature(self, valid_loader):
        self.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = rein_forward(self.model, inputs)
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal Temperature for Model: {self.temperature.item():.3f}")
        return self


class EnsembleWithIndividualTS(nn.Module):
    """
    개별 모델에 TS 적용 후 앙상블하는 클래스
    """
    def __init__(self, models, device='cuda:0'):
        super(EnsembleWithIndividualTS, self).__init__()
        self.models = nn.ModuleList([ModelWithTemperature(model, device) for model in models])
        self.device = device

    def set_temperature(self, valid_loader):
        """
        각 모델별 Temperature 찾기
        """
        for model in self.models:
            model.set_temperature(valid_loader)

    def forward(self, inputs):
        """
        개별 모델 TS 적용 후 Ensemble
        """
        logits_list = []
        for model in self.models:
            logits = model(inputs)  # 각 모델은 이미 TS가 적용됨
            logits_list.append(logits)

        ensemble_logits = torch.stack(logits_list).mean(dim=0)  # 평균 ensemble logits
        return ensemble_logits


# class EnsembleWithTemperature(nn.Module):
#     """
#     Temperature Scaling을 Ensemble Logits에 적용하는 클래스
#     """
#     def __init__(self, models, device='cuda:0'):
#         super(EnsembleWithTemperature, self).__init__()
#         self.models = models  # 여러 개의 모델 리스트
#         self.temperature = nn.Parameter(torch.ones(1) * 1.0)
#         self.device = device

#     def forward(self, inputs):
#         """
#         Ensemble logits 계산 및 Temperature Scaling 적용
#         """
#         logits_list = []
#         for model in self.models:
#             logits = rein_forward(model, inputs)  # 개별 모델에서 logits 추출
#             logits_list.append(logits)
        
#         ensemble_logits = torch.stack(logits_list).mean(dim=0)  # 평균 ensemble logits
#         return self.temperature_scale(ensemble_logits)

#     def temperature_scale(self, logits):
#         """
#         Temperature Scaling 적용
#         """
#         return logits / self.temperature.to(logits.device)

#     def set_temperature(self, valid_loader):
#         """
#         Validation Set을 이용하여 최적 Temperature 찾기
#         """
#         self.eval()
#         nll_criterion = nn.CrossEntropyLoss().to(self.device)
#         ece_criterion = _ECELoss().to(self.device)

#         logits_list = []
#         labels_list = []
#         with torch.no_grad():
#             for inputs, labels in valid_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 logits = self.forward(inputs)
#                 logits_list.append(logits.cpu())
#                 labels_list.append(labels.cpu())

#         logits = torch.cat(logits_list)
#         labels = torch.cat(labels_list)

#         # Temperature Scaling 전의 NLL 및 ECE 계산
#         before_temperature_nll = nll_criterion(logits, labels).item()
#         before_temperature_ece = ece_criterion(logits, labels).item()
#         print(f"Before Temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}")

#         # 최적 Temperature 학습 (NLL 최소화)
#         optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

#         def eval():
#             optimizer.zero_grad()
#             loss = nll_criterion(self.temperature_scale(logits), labels)
#             loss.backward()
#             return loss

#         optimizer.step(eval)

#         # Temperature Scaling 후의 NLL 및 ECE 계산
#         after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
#         after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
#         print(f"Optimal Temperature: {self.temperature.item():.3f}")
#         print(f"After Temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}")

#         return self

#     def get_temperature(self):
#         return self.temperature
