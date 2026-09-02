import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit, UniAttentionUnit
import torch.nn.functional as F

class Efin(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,
                 base_hidden_dims=[100,100,100,100],base_hidden_func = torch.nn.ELU(),output_activation_base=torch.nn.Sigmoid(),
                 lift_hidden_dims=[100,100,100,100],lift_hidden_func = torch.nn.ELU(),output_activation_lift=torch.nn.Sigmoid(),
                 ipw_hidden_dims=[100,100,100,100],ipw_hidden_func = torch.nn.ELU(),output_activation_ipw=torch.nn.Sigmoid(),
                 attention_dim = 32,
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Efin',device='cpu'):
        super(Efin, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)

        # self-attention
        self.self_attention = UniAttentionUnit(out_dim=attention_dim)

        # cross-attention
        self.cross_attention = UniAttentionUnit(out_dim=attention_dim)

        # treatment tower
        for treatment_label in self.treatment_label_list:
            if treatment_label == 0:
                self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = input_dim*attention_dim, 
                 hidden_dims=base_hidden_dims, 
                 share_output_dim=None, 
                 activation=base_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task=task, 
                 classi_nums=classi_nums, 
                 output_activation=output_activation_base,
                 device=device, 
                 use_xavier=True)
            else:
                self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = input_dim*attention_dim, 
                    hidden_dims=lift_hidden_dims, 
                    share_output_dim=None, 
                    activation=lift_hidden_func, 
                    use_batch_norm=True, 
                    use_dropout=True, 
                    dropout_rate=0.3, 
                    task=task, 
                    classi_nums=classi_nums, 
                    output_activation=output_activation_lift,
                    device=device, 
                    use_xavier=True)
        
         # ipw tower
        self.ipw_tower = TowerUnit(input_dim = input_dim*attention_dim, 
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


    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        t = t.squeeze()
        t = F.one_hot(t.to(torch.long), num_classes=len(self.treatment_label_list)).float()

        # self-attention
        self_att_out,self_weight = self.self_attention(x_q = x.unsqueeze(-1),x_kv=None)
        dims = self_att_out.shape
        self_att_out = torch.reshape(self_att_out, (dims[0], dims[1] * dims[2]))
        
        # corss-attention
        cross_att_out,cross_weight = self.cross_attention(x_q = x.unsqueeze(-1),x_kv=t.unsqueeze(-1))
        dims = cross_att_out.shape
        cross_att_out = torch.reshape(cross_att_out, (dims[0], dims[1] * dims[2]))

        pre = []
        ate = []

        predcit_pro_0 = None
        for treatment_label in self.treatment_label_list:
            if treatment_label == 0:
                predcit_pro_0 = self.treatment_model[str(treatment_label)](self_att_out).squeeze().unsqueeze(1)
                pre.append(predcit_pro_0)
            else:
                predcit_pro = self.treatment_model[str(treatment_label)](cross_att_out).squeeze().unsqueeze(1)
                pre.append(predcit_pro_0.detach() + predcit_pro)
                ate.append(predcit_pro)

        # ipw
        ipw = self.ipw_tower(cross_att_out)
        pre.append(ipw)

        return torch.cat(ate, dim=1) if len(ate) !=0 else None,pre,None


def efin_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None,X_true=None,**kwargs):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = torch.gather(torch.cat(y_preds[:-1], dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)
    ipw = y_preds[-1]

    y_true_dict = {}
    y_pred_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]

    # loss ipw
    if len(treatment_label_list) == 2:
        ipw_criterion = nn.BCEWithLogitsLoss()
        loss_ipw = ipw_criterion(ipw, t.float())

    if len(treatment_label_list) > 2: 
        ipw_criterion = nn.CrossEntropyLoss()
        loss_ipw = ipw_criterion(ipw, t.squeeze())

    # 计算每个treatment的损失
    if task == 'classification':
        if loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type =='BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("loss_type must be 'BCEWithLogitsLoss' or 'BCELoss'")
    elif task == 'regression':
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss(reduction='mean')
        elif loss_type =='huberloss':
            criterion = nn.SmoothL1Loss() 
        else:
            raise ValueError("loss_type must be 'mse' or 'huberloss'")
    else:
        raise ValueError("task must be 'classification' or'regression'")
    
    loss_treat = 0
    for treatment in treatment_label_list:
        loss_treat += criterion(y_pred_dict[treatment], y_true_dict[treatment])
    loss = loss_treat + loss_ipw
    return loss, loss_treat, loss_ipw
