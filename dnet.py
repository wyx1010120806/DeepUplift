import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Dnet(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],
                 base_hidden_dims=[100,100,100,100],base_share_dim=64,
                 ipw_hidden_dims=[64,64,64,64,64],output_activation_ipw=torch.nn.Sigmoid(),
                 quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(),ipw_hidden_func = torch.nn.ELU(),
                 task = 'regression',classi_nums=1, treatment_label_list=[0,1,2,3],model_type='Dnet',device='cpu'):
        super(Dnet, self).__init__()
        if task is None or task == "classification":
            raise ValueError("task must be 'regression'")
        if max(quantiles) >= 1 or min(quantiles) <= 0:
            raise ValueError("quantiles must be between 0 and 1")
        
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.ncq_value_layer = nn.ModuleDict()
        self.ncq_delta_layer = nn.ModuleDict()
        self.device = device
        self.quantiles = quantiles
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)
        
        # share tower
        self.share_tower = TowerUnit(input_dim = input_dim, 
                 hidden_dims=share_hidden_dims, 
                 share_output_dim=share_dim, 
                 activation=share_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='share', 
                 classi_nums=None, 
                 device=device, 
                 use_xavier=True)

        for treatment_label in self.treatment_label_list:
            # treatment tower
            self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = share_dim, 
                 hidden_dims=base_hidden_dims, 
                 share_output_dim=base_share_dim, 
                 activation=base_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='share', 
                 classi_nums=None, 
                 device=device, 
                 use_xavier=True)
        
        # ipw tower
        self.ipw_tower = TowerUnit(input_dim = share_dim, 
                 hidden_dims=ipw_hidden_dims, 
                 share_output_dim=None, 
                 activation=ipw_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='classification', 
                 classi_nums=self.treatment_nums, 
                 output_activation=output_activation_ipw,
                 device=device, 
                 use_xavier=True)

        # Ncq value layer
        for treatment_label in self.treatment_label_list:
            self.ncq_value_layer[str(treatment_label)] = TowerUnit(input_dim = base_share_dim, 
                 hidden_dims=None, 
                 share_output_dim=None, 
                 activation=None, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task=task, 
                 classi_nums=classi_nums, 
                 output_activation=None,
                 device=device, 
                 use_xavier=True)
        
        # Ncq delta layer
        for treatment_label in self.treatment_label_list:
            self.ncq_delta_layer[str(treatment_label)] = TowerUnit(input_dim = base_share_dim, 
                 hidden_dims=None, 
                 share_output_dim=len(quantiles), 
                 activation=torch.nn.ELU(), 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='share', 
                 classi_nums=None, 
                 device=device, 
                 use_xavier=True)

    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        # print(f'输入{x}')

        #Base net
        share_out = self.share_tower(x)
    
        # T-tower
        ipw = self.ipw_tower(share_out)

        #R-tower
        pre = []
        ate = []
        base = None
        for treatment_label in self.treatment_label_list:
            treatment_out = self.treatment_model[str(treatment_label)](share_out)
            value = self.ncq_value_layer[str(treatment_label)](treatment_out)
            delta = self.ncq_delta_layer[str(treatment_label)](treatment_out)
           
            K = delta.shape[1]
            w = torch.arange(K, 0, -1, device=self.device)
            weighted = delta*w
            s = weighted.sum(dim=1)/K
            delta_ = torch.cumsum(delta, dim=1) - s.squeeze().unsqueeze(1)
            value_ = value + delta_
        
            pre.append(value_)
            if treatment_label == 0:
                base = value_.mean(dim=1).squeeze().unsqueeze(1)
            else:
                ate.append(value_.mean(dim=1).squeeze().unsqueeze(1) - base)
            
        pre.append(ipw)
        return torch.cat(ate, dim=1),pre,None

def multi_quantile_loss(y_pred_q, y_true,quantiles):
    """
    y_true:   [B]
    y_pred_q: [B, K]，按 quantiles 的顺序输出的各分位预测
    quantiles: list/tuple 长度 K，如 [0.1, 0.5, 0.9]
    返回:     标量 loss（对所有分位平均）
    """
    y_true = y_true.view(-1, 1).expand_as(y_pred_q)  # [B, K]
    qs = torch.tensor(quantiles, device=y_pred_q.device, dtype=y_pred_q.dtype)  # [K]
    err = y_true - y_pred_q                         # [B, K]
    loss = torch.maximum(qs * err, (qs - 1) * err)  # [B, K]
    return loss.mean()


def dnet_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None,X_true=None,**kwargs):
    if task is None or task == "classification":
        raise ValueError("task must be 'regression'")
    quantiles = kwargs.get('quantiles')
    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = y_preds[:-1]

    ipw = y_preds[-1]

    y_true_dict = {}
    y_pred_dict = {}
    i = 0
   
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[i][mask.squeeze(1), :]
        i += 1

    # 计算ipw损失
    if len(treatment_label_list) == 2:
        ipw_criterion = nn.BCEWithLogitsLoss()
        loss_ipw = ipw_criterion(ipw, t.float())

    if len(treatment_label_list) > 2: 
        ipw_criterion = nn.CrossEntropyLoss()
        loss_ipw = ipw_criterion(ipw, t.squeeze())

    # 计算每个treatment的损失
    loss_treat = 0
    for treatment in treatment_label_list:
        loss_treat += multi_quantile_loss(y_pred_dict[treatment], y_true_dict[treatment],quantiles)
    loss = loss_treat + loss_ipw
    return loss, loss_treat, loss_ipw



        
        