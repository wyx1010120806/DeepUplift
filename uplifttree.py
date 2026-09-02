from causalml.inference.tree import UpliftTreeClassifier,UpliftRandomForestClassifier,CausalTreeRegressor,CausalRandomForestRegressor
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from IPython.display import Image
from causalml.inference.tree.plot import plot_causal_tree
import matplotlib.pyplot as plt
import pandas as pd
from econml.dml import CausalForestDML
import xgboost as xgb
from lightgbm import LGBMClassifier,LGBMRegressor

class UpliftTreeModel():
    def __init__(self,task,model_type,treatment_list,features_list,**params):
        self.task = task
        self.model_type = model_type
        self.treatment_list = treatment_list
        self.features_list = features_list
        self.model = None
        self.params = params
        
    def fit(self,X,y,treatment):
        treatment=treatment.astype(int).astype(str)
        if self.task == 'classification':
            if self.model_type == 'tree':
                self.model = UpliftTreeClassifier(control_name='0',**self.params)
                self.model.fit(X=X,y=y,treatment=treatment)
            elif self.model_type == 'forest':
                self.model = UpliftRandomForestClassifier(control_name='0',**self.params)
                self.model.fit(X=X,y=y,treatment=treatment)
            elif self.model_type == 'causalforestdml':
                self.model = CausalForestDML(model_y=LGBMClassifier(verbosity=-1,n_estimators = 100,max_depth=5,objective='binary',n_jobs=-1),
                                             model_t=LGBMClassifier(verbosity=-1,n_estimators = 100,max_depth=5,objective='multiclass' if len(self.treatment_list) > 2 else 'binary',n_jobs=-1),
                                             discrete_outcome=True,
                                             discrete_treatment=True,
                                             subforest_size=1,
                                             inference = False,
                                             n_jobs = 2,
                                             **self.params
                                             )
                self.model.fit(X=X,Y=y,T=treatment)
            else:
                raise ValueError("model_type must be 'tree' or 'forest'")
        elif self.task == 'regression' and len(self.treatment_list) == 2:
            if self.model_type == 'tree':
                self.model = CausalTreeRegressor(control_name='0',**self.params)
                self.model.fit(X=X,y=y,treatment=treatment)
            elif self.model_type == 'forest':
                self.model = CausalRandomForestRegressor(control_name='0',**self.params)
                self.model.fit(X=X,y=y,treatment=treatment.astype(int))
            elif self.model_type == 'causalforestdml':
                self.model = CausalForestDML(model_y=LGBMRegressor(verbosity=-1,n_estimators = 100,max_depth=5,n_jobs=-1),
                                             model_t=LGBMClassifier(verbosity=-1,n_estimators = 100,max_depth=5,objective='multiclass' if len(self.treatment_list) > 2 else 'binary',n_jobs=-1),
                                             discrete_outcome=False,
                                             discrete_treatment=True,
                                             subforest_size=1,
                                             inference = False,
                                             n_jobs = 2,
                                             **self.params
                                             )
                self.model.fit(X=X,Y=y,T=treatment)
            else:
                raise ValueError("model_type must be 'tree' or 'forest'")
        elif self.task == 'regression' and len(self.treatment_list) > 2:
            if self.model_type == 'causalforestdml':
                self.model = CausalForestDML(model_y=LGBMRegressor(verbosity=-1,n_estimators = 100,max_depth=5,n_jobs=-1),
                                             model_t=LGBMClassifier(verbosity=-1,n_estimators = 100,max_depth=5,objective='multiclass' if len(self.treatment_list) > 2 else 'binary',n_jobs=-1),
                                             discrete_outcome=False,
                                             discrete_treatment=True,
                                             subforest_size=1,
                                             inference = False,
                                             n_jobs = 2,
                                             **self.params
                                             )
                self.model.fit(X=X,Y=y,T=treatment)
            else:
                self.model = {}

                for treatment_name in self.treatment_list:
                    if treatment_name == 0:
                        continue

                    mask = (treatment == '0') | (treatment == str(treatment_name))   # 选择 treatment 为 0 或 1 的行
                    X_sub = X[mask]
                    y_sub = y[mask]
                    t_sub = treatment[mask]

                    if self.model_type == 'tree':
                        model = CausalTreeRegressor(control_name='0',**self.params)
                        model.fit(X=X_sub,y=y_sub,treatment=t_sub)
                    elif self.model_type == 'forest':
                        model = CausalRandomForestRegressor(control_name=0,**self.params)
                        model.fit(X=X_sub,y=y_sub,treatment=t_sub.astype(int))
                    else:
                        raise ValueError("model_type must be 'tree' or 'forest'")
                    
                    self.model[treatment_name] = model
        else:
            raise ValueError("task must be 'regression' or 'classification'")
        
    def predict(self,X):
        if self.model_type == 'causalforestdml':
            predictions = {}
            for each in self.treatment_list[1:]:
                predictions[each] = self.model.effect(X,T0=str(self.treatment_list[0]),T1=str(each)).squeeze(-1) if self.task == 'classification' else self.model.effect(X,T0=str(self.treatment_list[0]),T1=str(each))
            return  pd.DataFrame(predictions)
        
        if len(self.treatment_list) == 2 or self.task == 'classification':
            if self.task == 'classification' and self.model_type == 'forest':
                cols = []
                for each in self.treatment_list[1:]:
                    cols.append(f'delta_{each}')
                res = self.model.predict(X, full_output=True)[cols]
                for each in self.treatment_list[1:]:
                    res = res.rename(columns={f'delta_{each}':f'{each}'})
                return res
            else:
                if self.task == 'classification' and self.model_type == 'tree':
                    arr = self.model.predict(X)
                    diff = arr[:, 1:] - arr[:, [0]]  
                    col_names = [str(i) for i in range(1, arr.shape[1])]
                    return pd.DataFrame(diff, columns=col_names)
                else:
                    return pd.DataFrame(self.model.predict(X), columns=[1])
        else:
            predictions = {}
            for treatment_name in self.treatment_list:
                if treatment_name == 0:
                    continue
                predictions[treatment_name] = self.model[treatment_name].predict(X)
            return  pd.DataFrame(predictions)
    
    # def visualize(self):
    #     if len(self.treatment_list) == 2 or self.task == 'classification':
    #         print('***************字符串形式展示***************')
    #         result = uplift_tree_string(self.model.fitted_uplift_tree, self.features_list)
    #         print(result)
    #         print('***************树图形展示***************')
    #         graph = uplift_tree_plot(self.model.fitted_uplift_tree, self.features_list)
    #         print(graph)
    #     else:
    #         for treatment_name in self.treatment_list:
    #             if treatment_name == 0:
    #                 continue
    #             print(f'***************{treatment_name}***************')
    #             print('***************字符串形式展示***************')
    #             result = uplift_tree_string(self.model[treatment_name].fitted_uplift_tree, self.features_list)
    #             print(result)
    #             print('***************树图形展示***************')
    #             graph = uplift_tree_plot(self.model[treatment_name].fitted_uplift_tree, self.features_list)
    #             Image(graph.create_png())
                

