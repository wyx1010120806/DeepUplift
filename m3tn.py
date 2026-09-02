import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class M3tn(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,expert_dim=6,
                 expert_hidden_dims =[[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64],[64,64,64,64,64]],expert_hidden_func = torch.nn.ELU(),
                 gate_hidden_dims =[[128],[128],[128],[128]],gate_hidden_func = torch.nn.ELU(),
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU(),
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Tarnet',device='cpu'):
        super(M3tn, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.expert = nn.ModuleList()
        self.gate = nn.ModuleDict()
        self.treatment_model = nn.ModuleDict()
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
        for treatment_label in self.treatment_label_list:
            self.gate[str(treatment_label)] = TowerUnit(input_dim = input_dim, 
                 hidden_dims=gate_hidden_dims[treatment_label], 
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


        for treatment_label in self.treatment_label_list:
            # treatment tower
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

        # 专家网络
        expert_out_list = []
        for expert_ in self.expert:
            expert_out_list.append(expert_(x))
        expert_out = torch.stack(expert_out_list, dim=1)

        pre = []
        ate = []

        # 门网络+输入塔
        for treatment_label in self.treatment_label_list:
            gate_out = self.gate[str(treatment_label)](x)
            predict_input = (expert_out * gate_out[:, :, None]).sum(dim=1)

            predcit_pro = self.treatment_model[str(treatment_label)](predict_input).squeeze().unsqueeze(1)
            pre.append(predcit_pro)
            if treatment_label !=0:
                ate.append(predcit_pro)

        return torch.cat(ate, dim=1) if len(ate) !=0 else None,pre,None

def m3tn_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None,X_true=None,**kwargs):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_0 = y_preds[0].squeeze().unsqueeze(1)
    y_pred = torch.gather(torch.cat(y_preds, dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)

    y_true_dict = {}
    y_pred_dict = {}
    y_0_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]
        if treatment != 0:
            y_0_dict[treatment] = y_0[mask]

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
        if treatment == 0:
            loss_treat += criterion(y_pred_dict[treatment], y_true_dict[treatment])
        else:
            loss_treat += criterion(y_0_dict[treatment] + y_pred_dict[treatment], y_true_dict[treatment])
    loss = loss_treat
    return loss, loss_treat, None



        
        