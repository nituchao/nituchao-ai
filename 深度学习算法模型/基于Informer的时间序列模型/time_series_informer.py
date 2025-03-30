import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context
        context = self._get_initial_context(values, L_Q)

        # update the context with selected top_k queries
        scores_top = self.dropout(torch.softmax(scores_top, dim=-1))

        context_in = torch.matmul(scores_top, values)

        context_in = context_in.transpose(2,1)
        
        return context_in

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = x.transpose(1,2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1,2)

        return self.norm2(x+y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.0, attn='prob', activation='gelu', 
                output_attention = False):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Attention
        Attn = ProbAttention if attn=='prob' else None
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.pos_encoder(enc_out)
        
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out = self.projection(enc_out)
        
        return dec_out

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.data = torch.FloatTensor(data)
        self.input_window = input_window
        self.output_window = output_window
        
    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_window]
        y = self.data[idx+self.input_window:idx+self.input_window+self.output_window]
        return x, y

def train_model():
    # 生成示例数据
    t = np.arange(0, 1000, 0.1)
    data = np.sin(t) + np.random.normal(0, 0.1, len(t))
    
    # 数据预处理
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # 设置参数
    input_window = 100
    output_window = 20
    batch_size = 32
    
    # 创建数据集和数据加载器
    dataset = TimeSeriesDataset(data_scaled, input_window, output_window)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=input_window,
        label_len=output_window,
        out_len=output_window,
        factor=5,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        dropout=0.1
    )
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model, scaler

def predict(model, scaler, input_data, prediction_steps):
    model.eval()
    with torch.no_grad():
        # 准备输入数据
        scaled_input = scaler.transform(input_data.reshape(-1, 1))
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)
        
        # 预测
        output = model(input_tensor)
        
        # 转换回原始比例
        predictions = scaler.inverse_transform(output.squeeze().numpy().reshape(-1, 1))
        
    return predictions

if __name__ == "__main__":
    # 训练模型
    model, scaler = train_model()
    
    # 生成测试数据
    test_input = np.sin(np.arange(0, 10, 0.1))
    
    # 预测
    predictions = predict(model, scaler, test_input, 20)
    
    # 打印结果
    print("Predictions shape:", predictions.shape)
    print("First few predictions:", predictions[:5].flatten()) 