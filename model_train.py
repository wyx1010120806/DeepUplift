#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import pandas as pd
import torch
import pickle
from torch.nn import *
from urllib.parse import urlparse, urlunparse
from pathlib import Path
# from uplifttree import *
from functools import partial
from esn_tarnet import *
from feature_select import *
from s_learner import *
from t_learner import *
from tarnet import *
from dragonnet import *
from m3tn import *
from dnet import * 
from efin import *
from mtst import *
from x_learner import *
from descn import *
from cfrnet import *
import subprocess
import json

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



# In[ ]:


def run_shell(hdfs_ls_cmd):
    print(hdfs_ls_cmd)
    result = subprocess.run(hdfs_ls_cmd, shell=True, capture_output=True, text=True)     
    if result.returncode != 0:
        print(f"Error listing files in HDFS: {result.stderr}")
    print(result)
    print(result.returncode)



try:
    # 读取JSON文件并转换为字典
    with open('/opt/tiger/rh2_params/custom_template_vars.json', 'r') as file:
        data = json.load(file)

    # 打印字典
    print(data)

    if "selected_features_path" in data.keys():
        feature_path = data['selected_features_path']
    else:
        feature_path = data['feature_path']

    feature_path = data['feature_path']
    feature = Path(feature_path).name
    p = urlparse(feature_path)
    parent_path = p.path.rsplit('/', 1)[0]  
    selected_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))

    feature_list_discrete_path = os.path.basename(feature_path).split('.')[0] + '_discrete.pkl'
    feature_list_discrete_hdfs_path = selected_path + '/' + feature_list_discrete_path
    discrete_size_cols_path = os.path.basename(feature_path).split('.')[0] + '_discrete_size.pkl'
    discrete_size_cols_hdfs_path = selected_path + '/' + discrete_size_cols_path

    train_data_path = data['train_data_path']
    train_data_path_name = Path(train_data_path).name

    y = data['y']
    treatment_col = data['treatment_col']

    s = data.get('treatment_label_list', '')
    treatment_label_list = [int(x.strip()) for x in s.split(',') if x.strip()]

    task = data['task']
    loss_type = data['loss_type']
    model_type = data['model_type']
    model_params = data['model_params']
    model_save_path = data['model_save_path']
    model_save_name = Path(model_save_path).name
    p = urlparse(model_save_path)
    parent_path = p.path.rsplit('/', 1)[0]  
    model_save_name_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))



    epochs = int(data['epochs'])
    batch_size = int(data['batch_size'])
    learning_rate = float(data['learning_rate'])

except Exception as e:
    # 跳过：记录错误并继续
    print(f"跳过：{e}")


# In[ ]:
try:
    # 读取JSON文件并转换为字典
    with open('/opt/tiger/rh2_params/data_params.json', 'r') as file:
        data = json.load(file)

    # 打印字典
    print(data)
    if "selected_features_path" in data.keys():
        feature_path = data['selected_features_path']
    else:
        feature_path = data['feature_path']
        
    feature = Path(feature_path).name
    p = urlparse(feature_path)
    parent_path = p.path.rsplit('/', 1)[0]  
    selected_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))

    feature_list_discrete_path = os.path.basename(feature_path).split('.')[0] + '_discrete.pkl'
    feature_list_discrete_hdfs_path = selected_path + '/' + feature_list_discrete_path
    discrete_size_cols_path = os.path.basename(feature_path).split('.')[0] + '_discrete_size.pkl'
    discrete_size_cols_hdfs_path = selected_path + '/' + discrete_size_cols_path

    train_data_path = data['train_data_path']
    train_data_path_name = Path(train_data_path).name

    y = data['y']
    treatment_col = data['treatment_col']

    s = data.get('treatment_label_list', '')
    treatment_label_list = [int(x.strip()) for x in s.split(',') if x.strip()]

    task = data['task']
    loss_type = data['loss_type']
    model_type = data['model_type']
    model_params = data['model_params']
    model_save_path = data['model_save_path']
    model_save_name = Path(model_save_path).name
    p = urlparse(model_save_path)
    parent_path = p.path.rsplit('/', 1)[0]  
    model_save_name_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))

except Exception as e:
    # 跳过：记录错误并继续
    print(f"跳过：{e}")




try:
    # 读取JSON文件并转换为字典
    with open('/opt/tiger/rh2_params/train_params.json', 'r') as file:
        train_params = json.load(file)
    # 打印字典
    print(train_params)

    model_select_type = train_params['model_select_type']
    print(model_select_type,model_type)
    if model_select_type in model_type:
        idx = model_type.index(model_select_type)
        model_type = model_select_type
        model_params = model_params[idx]
        model_save_path = model_save_path[idx]
        model_save_name = Path(model_save_path).name
        p = urlparse(model_save_path)
        parent_path = p.path.rsplit('/', 1)[0]  
        model_save_name_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))
        print(model_type,model_params,model_save_path)
    else:
        print("未设置当前模型")
        import sys
        sys.exit(0) 
    
    epochs = int(train_params['epochs'])
    batch_size = int(train_params['batch_size'])
    learning_rate = float(train_params['learning_rate'])

except Exception as e:
    # 跳过：记录错误并继续
    print(f"跳过：{e}")


if task == 'classification' and model_type == 'dnet':
    print("dnet不支持分类任务,退出运行")
    import sys
    sys.exit(0) 


# In[ ]:


run_shell(f"rm -r ./{train_data_path_name}")
run_shell(f"hdfs dfs -get {train_data_path} ./")

run_shell(f"rm -r ./{feature}")
run_shell(f"hdfs dfs -get {feature_path} ./")

run_shell(f"rm -r ./{feature_list_discrete_path}")
run_shell(f"hdfs dfs -get {feature_list_discrete_hdfs_path} ./")

run_shell(f"rm -r ./{discrete_size_cols_path}")
run_shell(f"hdfs dfs -get {discrete_size_cols_hdfs_path} ./")


# In[ ]:


#读取特征列表
with open(f"./{feature}", 'rb') as f:
    feature_list = pickle.load(f)
print(len(feature_list))

with open(f"./{feature_list_discrete_path}", 'rb') as f:
    feature_list_discrete = pickle.load(f)
print(len(feature_list_discrete))

with open(f"./{discrete_size_cols_path}", 'rb') as f:
    discrete_size_cols = pickle.load(f)
print(len(discrete_size_cols))


# In[ ]:


if not model_params:
    if model_type == 'tarnet':
        model_params = dict(
            embedding_dim=3,share_dim=64,
            share_hidden_dims =[256,128,128,64],
            base_hidden_dims = [64,32,32,16],output_activation_base=None,
            share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'esn_tarnet':
        model_params = dict(
            embedding_dim=3,share_dim=64,
            share_hidden_dims =[256,128,64,64],
            base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),
            ipw_hidden_dims=[256,128,64,64],output_activation_ipw=None,
            share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'slearner':
        model_params = dict(
            embedding_dim=3,
            base_hidden_dims=[64,32,32,16],output_activation_base=None,base_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'tlearner':
        model_params = dict(
            embedding_dim=3,
            base_hidden_dims=[64,32,32,16],output_activation_base=None,base_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'xlearner':
        model_params = dict(
            embedding_dim=3,
            base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU()
            ,lift_hidden_dims=[64,32,32,16],lift_activation_base=None,lift_hidden_func = torch.nn.ELU()
            ,treatment_label_weight=[0.25,0.25,0.25,0.25],lift_weight=[0.5,0.5,0.5,0.5]
        )
    elif model_type == 'descn':
        model_params = dict(
            embedding_dim=3,share_dim=64,
                    share_hidden_dims =[256,128,64,64],
                    base_hidden_dims=[64,32,32,16],output_activation_base=None,
                    ipw_hidden_dims=[64,32,32,16],output_activation_ipw=None,
                    pseudo_hidden_dims=[64,32,32,16],output_activation_pseudo=None,
                    share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU(),pseudo_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'dragonnet':
        model_params = dict(
            embedding_dim=3,share_dim=64,
            share_hidden_dims =[512,256,256,128], share_hidden_func = torch.nn.ELU(),
            base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU(),
            ipw_hidden_dims=[64,32,32,16],output_activation_ipw=torch.nn.Sigmoid(),ipw_hidden_func = torch.nn.ELU(),
            epsilons_hidden_dims=[64,32,32,16],output_activation_epsilons=torch.nn.Sigmoid(),epsilons_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'cfrnet':
        model_params = dict(
            embedding_dim=3,share_dim=64,
            share_hidden_dims =[256,128,128,64],
            base_hidden_dims = [64,32,32,16],output_activation_base=None,
            share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'dnet':
        model_params = dict(
            embedding_dim=3,share_dim=128,
            share_hidden_dims =[512,256,256,128],
            base_hidden_dims=[64,32,32,16],base_share_dim=16,
            ipw_hidden_dims=[64,32,32,16],output_activation_ipw=None,
            quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        )
    elif model_type == 'm3tn':
        model_params = dict(
            embedding_dim=3,expert_dim=128,
            expert_hidden_dims =[[512,256,128],[512,256,128],[512,256,128],[512,256,128],[512,256,128]],expert_hidden_func = torch.nn.ELU(),
            gate_hidden_dims =[[256],[256],[256],[256]],gate_hidden_func = torch.nn.ELU(),
            base_hidden_dims=[128,64,64],output_activation_base=None,base_hidden_func = torch.nn.ELU()
        )
    elif model_type == 'efin':
        model_params = dict(
            embedding_dim=3,
            base_hidden_dims=[256,128,64],base_hidden_func = torch.nn.ELU(),output_activation_base=None,
            lift_hidden_dims=[256,128,64],lift_hidden_func = torch.nn.ELU(),output_activation_lift=None,
            ipw_hidden_dims=[256,128,64],ipw_hidden_func = torch.nn.ELU(),output_activation_ipw=None,
            attention_dim = 32
        )
    elif model_type == 'mtst':
        model_params = dict(
            embedding_dim=3,expert_dim=128,
            expert_hidden_dims =[[512,256,128],[512,256,128],[512,256,128],[512,256,128],[512,256,128]],expert_hidden_func = torch.nn.ELU(),
            gate_hidden_dims =[256],gate_hidden_func = torch.nn.ELU(),
            base_hidden_dims=[128,64,64],output_activation_base=None,base_hidden_func = torch.nn.ELU(),
            attention_dim = 32
        )
    elif model_type == 'uplifttree':
        model_params = dict(
            max_depth=5
        )
    elif model_type == 'upliftforest':
        model_params = dict(
            max_depth=5,n_estimators=100
        )
    elif model_type == 'causalforestdml':
        model_params = dict(
            max_depth=5,n_estimators=100
        )
    else:
        raise ValueError("model_type must be 'mtst', 'efin', 'm3tn', 'dnet', 'tarnet', 'cfrnet', 'esn_tarnet', 'slearner', 'tlearner', 'xlearner', 'descn', 'dragonnet', 'uplifttree','upliftforest','causalforestdml'")


# In[ ]:


if model_type == 'tarnet':
    model = Tarnet(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(tarnet_loss)
elif model_type == 'esn_tarnet':
    model = ESN_Tarnet(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(esn_tarnet_loss)
elif model_type == 'slearner':
    model = Slearner(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(slearn_loss)
elif model_type == 'tlearner':
    model = Tlearner(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(tlearn_loss)
elif model_type == 'xlearner':
    model = Xlearner(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(xlearn_loss)
elif model_type == 'descn':
    model = Descn(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(descn_loss)
elif model_type == 'dragonnet':
    model = Dragonnet(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(dragonnet_loss)
elif model_type == 'cfrnet':
    model = Cfrnet(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(cfrnet_loss)
elif model_type == 'dnet':
    model = Dnet(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(dnet_loss)
elif model_type == 'm3tn':
    model = M3tn(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(m3tn_loss)
elif model_type == 'efin':
    model = Efin(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(efin_loss)
elif model_type == 'mtst':
    model = Mtst(
        input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
        treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
        **model_params
    ).to(device)
    loss_f = partial(mtst_loss)
elif model_type == 'uplifttree':
    model = UpliftTreeModel(task=task,model_type='tree',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
elif model_type == 'upliftforest':
    model = UpliftTreeModel(task=task,model_type='forest',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
elif model_type == 'causalforestdml':
    model = UpliftTreeModel(task=task,model_type='causalforestdml',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
else:
    raise ValueError("model_type must be 'mtst', 'efin', 'm3tn', 'dnet','tarnet', 'cfrnet', 'esn_tarnet', 'slearner', 'tlearner', 'xlearner', 'descn', 'dragonnet', 'uplifttree','upliftforest','causalforestdml'")


# In[ ]:


df = pd.read_parquet(f"./{train_data_path_name}",columns=feature_list+[treatment_col,y]).fillna(0)

for column in df.columns:
    if df[column].dtype != 'float':
        df[column] = df[column].astype('float')


# In[ ]:

if model_type in ('uplifttree','upliftforest','causalforestdml'):
    X = df[feature_list].values
    y = df[y].values
    t = df[treatment_col].values
    model.fit(X, y, t)

    import pickle
    with open(f"./{model_save_name}", 'wb') as f:
        pickle.dump(model, f)

    run_shell(f"hdfs dfs -rm -r {model_save_path}")
    run_shell(f"hdfs dfs -mkdir {model_save_name_path}")
    run_shell(f"hdfs dfs -put ./{model_save_name} {model_save_name_path}")
        
else:
    model.fit(
        df=df,
        feature_list=feature_list,
        discrete_cols=feature_list_discrete,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_f=loss_f,
        tensorboard=False,
        num_workers=40,
        pin_memory=True,
        device=device,
        valid_perc=True,
        label_y=y,
        label_treatment=treatment_col,
        task=task,
        loss_type=loss_type,
        treatment_label_list=treatment_label_list,
        checkpoint_path = model_save_path,
        if_continued_train = 0,
        quantiles = model_params['quantiles'] if "quantiles" in model_params.keys() else None
    )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




