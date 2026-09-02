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
from m3tn import *
from dnet import * 
from t_learner import *
from tarnet import *
from dragonnet import *
from x_learner import *
from descn import *
from mtst import *
from efin import *
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


def run_shell(hdfs_ls_cmd):
    print(hdfs_ls_cmd)
    result = subprocess.run(hdfs_ls_cmd, shell=True, capture_output=True, text=True)     
    if result.returncode != 0:
        print(f"Error listing files in HDFS: {result.stderr}")
    print(result)
    print(result.returncode)

# 读取JSON文件并转换为字典
with open('/opt/tiger/rh2_params/custom_template_vars.json', 'r') as file:
    data = json.load(file)


# data = {
#     "predict_data_path": "hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/mall_predict_VN",
#     "feature_path": ["hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_feature_list_mall_gmv_int_label_selected.pkl",
#                      "hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_feature_list_mall_gmv_float_label_selected.pkl"
#                      ],
#     "treatment_label_list": ["0,1,2,3","0,1,2,3"],
#     "task": ["classification","regression"],
#     "model_type": ["xlearner","xlearner"],
#     "model_params": [{},{}],
#     "model_path": ["hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_mall_gmv_int_label_xlearner.pkl",
#                    "hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_mall_gmv_float_label_xlearner.pkl"
#                    ],
#     "output_path":"hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/mall_predict_VN_predict_result"
# }

if "selected_features_path" in data.keys():
    feature_path_list = data['selected_features_path']
else:
    feature_path_list = data['feature_path']
    

predict_data_path = data['predict_data_path']
predict_data_path_name = Path(predict_data_path).name

if "model_save_path" in data.keys():
    model_path_list = data['model_save_path']
else:
    model_path_list = data['model_path']

model_params_list = data['model_params']
model_type_list = data['model_type']
task_list = data['task']
treatment_label_all_list = data['treatment_label_list']

output_path = data['output_path']
output_path_name = Path(output_path).name
print(output_path_name)

if output_path == "hdfs://harunasg/home/byte_ecom_product_ds_sg" or output_path == "hdfs://harunasg/home/byte_ecom_product_ds_sg/":
    raise ValueError("严禁使用根目录，请指定更具体的子目录，以免误删")

begin = 0
end = 999999999

run_shell(f"rm -r ./{predict_data_path_name}")
run_shell(f"hdfs dfs -get {predict_data_path} ./")


import os
import warnings
warnings.filterwarnings("ignore")
# 定义文件夹路径
folder_path = f'./{predict_data_path_name}'

# 获取文件夹下的所有文件名
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print(len(file_names))
res = []
output_cols = []

df = pd.read_parquet(folder_path+'/'+file_names[0])

if 'user_id' in list(df.columns):
    output_cols.append('user_id')

if 'device_id' in list(df.columns):
    output_cols.append('device_id')

if 'country_code' in list(df.columns):
    output_cols.append('country_code')

if 'country' in list(df.columns):
    output_cols.append('country')

if 'user_active_country' in list(df.columns):
    output_cols.append('user_active_country')

output_cols = list(set(output_cols))
print(output_cols)




model_list = {}
for i in range(len(model_path_list)):
    model_type = model_type_list[i]
    model_path = model_path_list[i]
    model_params = model_params_list[i]
    task = task_list[i]
    treatment_label_list  = [int(x.strip()) for x in treatment_label_all_list[i].split(',') if x.strip()]

    feature_path = feature_path_list[i]
    feature = Path(feature_path).name
    p = urlparse(feature_path)
    parent_path = p.path.rsplit('/', 1)[0]  
    selected_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))

    feature_list_discrete_path = os.path.basename(feature_path).split('.')[0] + '_discrete.pkl'
    feature_list_discrete_hdfs_path = selected_path + '/' + feature_list_discrete_path
    discrete_size_cols_path = os.path.basename(feature_path).split('.')[0] + '_discrete_size.pkl'
    discrete_size_cols_hdfs_path = selected_path + '/' + discrete_size_cols_path

    run_shell(f"rm -r ./{feature}")
    run_shell(f"hdfs dfs -get {feature_path} ./")

    run_shell(f"rm -r ./{feature_list_discrete_path}")
    run_shell(f"hdfs dfs -get {feature_list_discrete_hdfs_path} ./")

    run_shell(f"rm -r ./{discrete_size_cols_path}")
    run_shell(f"hdfs dfs -get {discrete_size_cols_hdfs_path} ./")


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

    
    rank_ = 1
    print(rank_)
    import time
    for file_name in file_names[:1]:
        if file_name.endswith('.parquet') and begin <= int(file_name.split('-')[1]) and int(file_name.split('-')[1]) <= end:
            start_time = time.time()
            print(rank_,file_name)
            df_tmp = pd.read_parquet(folder_path+'/'+file_name).sample(n=10, random_state=42) 
            print(df_tmp.shape)

            uplift_predictions_cols = {}

            df = df_tmp.copy(deep=True)

            for column in feature_list:
                if df[column].dtype != 'float':
                    df[column] = df[column].astype('float')

            for i in range(len(feature_list_discrete)):
                df[feature_list_discrete[i]] = df[feature_list_discrete[i]].apply(lambda x: x if x >=0 and x <= discrete_size_cols[i]-2 else discrete_size_cols[i]-1)
            
            X_discrete = torch.tensor(df[feature_list_discrete].values, dtype=torch.float32).to(device)
            X_continuous = torch.tensor(df[[_ for _ in feature_list if _ not in feature_list_discrete]].values, dtype=torch.float32).to(device)
            target_length = df.shape[0]
            result_list = random.choices(treatment_label_list, k=target_length)
            t = torch.tensor(result_list, dtype=torch.int64).to(device)

            with torch.no_grad(): 
                uplift_predictions,y_preds,*eps = model(None, t, X_discrete=X_discrete, X_continuous=X_continuous)

    model_path_name = Path(model_path).name
    run_shell(f"rm -r ./{model_path_name}")
    run_shell(f"hdfs dfs -get {model_path} ./")

    checkpoint = torch.load(f"./{model_path_name}", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_list[model_path_name] = [model,feature_list,feature_list_discrete,discrete_size_cols]


# 打印所有文件名
rank_ = 1
print(rank_)
import time
for file_name in file_names:
    if file_name.endswith('.parquet') and begin <= int(file_name.split('-')[1]) and int(file_name.split('-')[1]) <= end:
        start_time = time.time()
        print(rank_,file_name)
        df_tmp = pd.read_parquet(folder_path+'/'+file_name)
        print(df_tmp.shape)

        uplift_predictions_cols = {}

        df = df_tmp.copy(deep=True)

        for k,v in model_list.items():
            k = Path(k).stem
            model = v[0]
            feature_list = v[1]
            feature_list_discrete = v[2]
            discrete_size_cols = v[3]

            for column in feature_list:
                if df[column].dtype != 'float':
                    df[column] = df[column].astype('float')

            for i in range(len(feature_list_discrete)):
                df[feature_list_discrete[i]] = df[feature_list_discrete[i]].apply(lambda x: x if x >=0 and x <= discrete_size_cols[i]-2 else discrete_size_cols[i]-1)
            
            X_discrete = torch.tensor(df[feature_list_discrete].values, dtype=torch.float32).to(device)
            X_continuous = torch.tensor(df[[_ for _ in feature_list if _ not in feature_list_discrete]].values, dtype=torch.float32).to(device)
            target_length = df.shape[0]
            result_list = random.choices(treatment_label_list, k=target_length)
            t = torch.tensor(result_list, dtype=torch.int64).to(device)

            model.eval()
            with torch.no_grad(): 
                uplift_predictions,y_preds,*eps = model(None, t, X_discrete=X_discrete, X_continuous=X_continuous)
            uplift_prediction = uplift_predictions.detach().cpu().numpy()
         
            for i in range(uplift_prediction.shape[1]):
                if rank_ == 1:
                    output_cols.append(str(i+1)+'_ITE_'+k)
                df[str(i+1)+'_ITE_'+k] = uplift_prediction[:,i]
            # print(df[output_cols])

        rank_ += 1
        print(f"predict time: {time.time() - start_time:.4f}s")

        res.append(df[output_cols])
        del df,df_tmp
        import gc
        gc.collect()

final_res = pd.concat(res)
final_res



final_res.to_parquet(f'./{output_path_name}.parquet', engine='pyarrow', index=False)


run_shell(f"hdfs dfs rm -r {output_path}")
run_shell(f"hdfs dfs -mkdir {output_path}")
run_shell(f"hdfs dfs -put -f ./{output_path_name}.parquet {output_path}")
