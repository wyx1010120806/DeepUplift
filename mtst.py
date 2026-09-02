import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit, UniAttentionUnit
import torch.nn.functional as F

class Mtst(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,
                 expert_dim=6,
                 expert_hidden_dims =[[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64]],expert_hidden_func = torch.nn.ELU(),
                 gate_hidden_dims =[128],gate_hidden_func = torch.nn.ELU(),
                 base_hidden_dims=[100,100,100,100],base_hidden_func = torch.nn.ELU(),output_activation_base=torch.nn.Sigmoid(),
                 attention_dim = 32,
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Mtst',device='cpu'):
        super(Mtst, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.expert = nn.ModuleList()
        self.cross_attention_second = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)

        # expert tower
        for layer in expert_hidden_dims:
            self.expert.append(TowerUnit(input_dim = input_dim, 
                    hidden_dims=layer, 
                    share_output_dim=expert_dim, 
                    activation=expert_hidden_func, 
                    use_batch_norm=True, 
                    use_dropout=True, 
                    dropout_rate=0.3, 
                    task='share', 
                    classi_nums=None, 
                    device=device, 
                    use_xavier=True))
        
        # gate tower
        self.gate = TowerUnit(input_dim = input_dim, 
                hidden_dims=gate_hidden_dims, 
                share_output_dim=None, 
                activation=gate_hidden_func, 
                use_batch_norm=True, 
                use_dropout=True, 
                dropout_rate=0.3, 
                task="classification", 
                classi_nums=len(expert_hidden_dims), 
                output_activation=None,
                device=device, 
                use_xavier=True)

        # cross-attention-base_treatment
        self.cross_attention_base = UniAttentionUnit(out_dim=attention_dim)

        # 有treatment反应基线
        self.is_treatment_model = TowerUnit(input_dim = 2*attention_dim, 
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
        
        # treatment tower & cross-attention-second_treatment
        for treatment_label in self.treatment_label_list:
            if treatment_label != 0:
                self.cross_attention_second[str(treatment_label)] = UniAttentionUnit(out_dim=attention_dim)
                self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = len(self.treatment_label_list)*attention_dim, 
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
                self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = expert_dim, 
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

    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        t = t.squeeze()
        t_bin = (t != 0).to(t.dtype)  
        t_bin = t_bin.squeeze()
        t_bin = F.one_hot(t_bin.to(torch.long), num_classes=2).float()
        t = F.one_hot(t.to(torch.long), num_classes=len(self.treatment_label_list)).float()

        # 专家网络
        expert_out_list = []
        for expert_ in self.expert:
            expert_out_list.append(expert_(x))
        expert_out = torch.stack(expert_out_list, dim=1)

        pre = []
        ate = []

        # 门网络
        gate_out = self.gate(x)
        gate_output = (expert_out * gate_out[:, :, None]).sum(dim=1)

        # 是否有treatment反应基线,交叉+mlp
        cross_att_out,cross_weight = self.cross_attention_base(x_q = t_bin.unsqueeze(-1),x_kv=gate_output.unsqueeze(-1))
        dims = cross_att_out.shape
        cross_att_out = torch.reshape(cross_att_out, (dims[0], dims[1] * dims[2]))
        is_treatment_output = self.is_treatment_model(cross_att_out)
        pre.append(is_treatment_output)

        # 无treatment反应
        no_treatment_output = self.treatment_model['0'](gate_output)
        pre.append(no_treatment_output)

        # 具体指定treatment反应,共享交叉+独立mlp
        for treatment_label in self.treatment_label_list:
            if treatment_label != 0:
                cross_att_out,cross_weight = self.cross_attention_second[str(treatment_label)](x_q = t.unsqueeze(-1),x_kv=gate_output.unsqueeze(-1))
                dims = cross_att_out.shape
                cross_att_out = torch.reshape(cross_att_out, (dims[0], dims[1] * dims[2]))
                treatment_output = self.treatment_model[str(treatment_label)](cross_att_out)
                pre.append(treatment_output)
                ate.append(is_treatment_output+treatment_output)
       
        return torch.cat(ate, dim=1) if len(ate) !=0 else None,pre,None


def mtst_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None,X_true=None,**kwargs):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred_0 = y_preds[1]
    y_pred_all_treatment = y_preds[0]
    y_pred = torch.gather(torch.cat(y_preds[1:], dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)

    y_true_dict = {}
    y_pred_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        if treatment == 0:
            y_pred_dict[treatment] = y_pred_0[mask]
        else:
            y_pred_dict[treatment] = y_pred_0[mask] + y_pred_all_treatment[mask] + y_pred[mask]

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
    loss = loss_treat 
    return loss, loss_treat, None
