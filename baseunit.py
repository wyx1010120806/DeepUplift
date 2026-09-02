import torch.nn as nn
import torch
import math
from typing import Optional, Tuple

class TowerUnit(nn.Module):
    def __init__(self, input_dim, hidden_dims=[], 
                 share_output_dim=16, activation=nn.ELU(), output_activation=None,
                 use_batch_norm=False, use_dropout=False, dropout_rate=0.2, 
                 task='share', classi_nums=None, 
                 device='cpu', use_xavier=True):
        """
        Tower unit for building multi-layer neural networks.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (list): List of hidden layer dimensions, default []
            share_output_dim (int): Output dimension for shared task, default 16
            activation (nn.Module): Activation function, default nn.ELU()
            use_batch_norm (bool): Whether to use batch normalization, default False
            use_dropout (bool): Whether to use dropout, default False
            dropout_rate (float): Dropout rate, default 0.2
            task (str): Task type ('share', 'classification', 'regression'), default 'share'
            classi_nums (int): Number of classes for classification task, default None
            device (str): Device for computation, default 'cpu'
            use_xavier (bool): Whether to use Xavier initialization, default True
        """
        super().__init__()
        self.device = device
        layers = []

        # hidden layers
        prev_dim = input_dim
        if hidden_dims:
            for dim in hidden_dims:
                linear_layer = nn.Linear(prev_dim, dim)
                if use_xavier:
                    nn.init.xavier_uniform_(linear_layer.weight)
                    if linear_layer.bias is not None:
                        nn.init.zeros_(linear_layer.bias)
                layers.append(linear_layer)
                if use_batch_norm:  
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(activation)
                if use_dropout: 
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = dim

        # output layers
        if task == 'classification' :
            if classi_nums == 2:
                output_dim , output_activation = 1 , output_activation 
            elif classi_nums > 2:
                output_dim , output_activation = classi_nums , torch.nn.Softmax(dim=1) 
            else:
                raise ValueError("classi_nums must be specified for classification task")
        elif task == 'regression':
            output_dim , output_activation = 1 ,output_activation   # No activation function for regression tasks
        elif task == 'share':
            output_dim , output_activation = share_output_dim , activation
        else:
            raise ValueError("task must be 'regression', 'classification' or 'share'")
            
        output_layer = nn.Linear(prev_dim, output_dim)
        if use_xavier:
            nn.init.xavier_uniform_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        if use_batch_norm:  
            layers.append(nn.BatchNorm1d(output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        """
        Forward propagation through the tower network.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size x input_dim]
            
        Returns:
            torch.Tensor: Output tensor with shape depending on task type
        """
        x = x.to(self.device)
        return self.net(x)



class UniAttentionUnit(nn.Module):
    """
    Unified Attention with lazy-initialized Q/K/V Linear layers.
    - self-attention: forward(x)               where Q=K=V=x
    - cross-attention: forward(x_q, x_kv=...)  where Q=x_q, K=V=x_kv
    Q_w/K_w/V_w will be created on first forward using input dims.
    """
    def __init__(self, out_dim: Optional[int] = None):
        """
        Args:
            out_dim: output feature dim for Q/K/V. If None, uses input dim.
                     For standard attention, out_dim == input_dim is fine.
        """
        super().__init__()
        # Placeholders; real layers will be created at first forward
        self.Q_w: Optional[nn.Linear] = None
        self.K_w: Optional[nn.Linear] = None
        self.V_w: Optional[nn.Linear] = None
        self.out_dim = out_dim
        self.softmax = nn.Softmax(dim=-1)

    def _build_linears(self, d_q_in: int, d_kv_in: int, device, dtype):
        # Decide output dims
        d_q_out = self.out_dim or d_q_in
        d_kv_out = self.out_dim or d_kv_in

        # Build and move to correct device/dtype
        self.Q_w = nn.Linear(d_q_in, d_q_out, bias=True).to(device=device, dtype=dtype)
        self.K_w = nn.Linear(d_kv_in, d_kv_out, bias=True).to(device=device, dtype=dtype)
        self.V_w = nn.Linear(d_kv_in, d_kv_out, bias=True).to(device=device, dtype=dtype)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_q:  (B, T_q, D_q) queries
            x_kv: (B, T_k, D_kv) keys/values source. If None, use x_q (self-attn)
        Returns:
            outputs:       (B, T_q, D_out)
            attn_weights:  (B, T_q, T_k)
        """
        if x_kv is None:
            x_kv = x_q  # self-attention

        B, T_q, D_q = x_q.shape
        _, T_k, D_kv = x_kv.shape

        # Lazy build on first call
        if self.Q_w is None or self.K_w is None or self.V_w is None:
            self._build_linears(D_q, D_kv, x_q.device, x_q.dtype)

        Q = self.Q_w(x_q)      # (B, T_q, D_q_out)
        K = self.K_w(x_kv)     # (B, T_k, D_kv_out)
        V = self.V_w(x_kv)     # (B, T_k, D_kv_out)

        # Require Q and K to have the same last dim for dot product
        if Q.size(-1) != K.size(-1):
            # Project K to Q's dim (or Q to K's dim), here we align K->Q
            K = nn.functional.linear(K, torch.eye(K.size(-1), Q.size(-1), device=K.device, dtype=K.dtype))
            # 或者更明确地增加一个适配层；为简洁起见此处用 F.linear+恒等映射占位

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))  # (B, T_q, T_k)
        attn_weights = self.softmax(scores)                                  # (B, T_q, T_k)
        outputs = torch.matmul(attn_weights, V)                               # (B, T_q, D_kv_out)
        return outputs, attn_weights

