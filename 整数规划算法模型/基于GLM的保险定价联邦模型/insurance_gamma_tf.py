# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow.compat.v1 as tf
import pandas as pd
import time

# %%
print(tf.__version__)

# %%
print(tf.test.is_gpu_available())

# %%
def train_test_split_fed(df_y, df_x, train_count = 10000):
    y_train = df_y[0:train_count]
    y_test = df_y[train_count+1:]
    x_train = df_x[0:train_count]
    x_test = df_x[train_count+1:]

    return y_train, y_test, x_train, x_test

# %%
df = pd.read_csv('data/insurance_claims_label_feature_fed.csv')

print('df shape: {}'.format(df.shape))
df = df[df['ClaimAmount'] > 0]
df = df[df['ClaimAmount'] < 5026.2015]
df.head()

# %%
columns_x = ['X{}'.format(idx) for idx in range(75)]
df_x = df[columns_x]

columns_y = ['AvgClaimAmount', 'ClaimNb']
df_y = df[columns_y]

print('df_x shape: {}'.format(df_x.shape))
df_x.head()

# %%
# from sklearn.model_selection import train_test_split

# y_train, y_test, x_train, x_test = train_test_split(df_y, df_x, random_state=0, shuffle=True, test_size=0.4)
y_train, y_test, x_train, x_test = train_test_split_fed(df_y, df_x, train_count=10000)

print('y_train shape: {}, \t y_test shape: {}, \n x_train shape: {}, \t x_test shape: {}'
      .format(y_train.shape, y_test.shape, x_train.shape, x_test.shape))

# %%
########################################################################
# 定义模型的网络结构
########################################################################

# 转换为张量
x_tensor = tf.constant(x_train, dtype=tf.float32)
y_tensor = tf.constant(y_train['AvgClaimAmount'], dtype=tf.float32)

# 定义模型参数
# weights = tf.Variable(tf.random.normal([75, 1]), dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
# bias = tf.Variable(tf.random.normal([1]), dtype=tf.float32, initializer=tf.zeros_initializer())

# weights = tf.get_variable('weights', shape=[75, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
weights = tf.get_variable('weights', shape=[75, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
bias = tf.get_variable('bias', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

# 定义伽马回归模型
def gamma_regression(X):
    logits = tf.matmul(X, weights) + bias
    return tf.exp(logits)   # 使用指数函数

# 定义损失函数
def gamma_loss(y_true, y_pred):
    return tf.reduce_mean((y_true / y_pred) - tf.math.log(y_true / y_pred) - 1)

# 定义优化器
# optimizer = tf.optimizers.Adam(learning_rate=0.01)
optimizer = tf.train.FtrlOptimizer(learning_rate=0.16921544485102483,
                                    l1_regularization_strength=1e-05,
                                    l2_regularization_strength=0.0005945795938393141,
                                    initial_accumulator_value=0.44352,
                                    learning_rate_power=-0.59496)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        y_pred = gamma_regression(x_tensor)
        loss = gamma_loss(y_tensor, y_pred)
        
    gradients = tape.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# 进行预测
pred_tf = gamma_regression(x_tensor)
print(pred_tf[:5].numpy())


# %%
# tf预测值
pred_tf = gamma_regression(x_tensor)
print(f'Train Prediction: {pred_tf[:5].numpy()}')

# %%
print(f'Train True: {y_train[:5]}')

# %%
y_test_pred = gamma_regression(tf.constant(x_test, dtype=tf.float32))
y_train_pred = gamma_regression(tf.constant(x_train, dtype=tf.float32))

# %%
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_train['AvgClaimAmount'], y_train_pred, sample_weight=y_train['ClaimNb'])
print("Train Mean Squared Error:", mse)

mse = mean_squared_error(y_test['AvgClaimAmount'], y_test_pred, sample_weight=y_test['ClaimNb'])
print("Test Mean Squared Error:", mse)

mse = mean_absolute_error(y_train['AvgClaimAmount'], y_train_pred, sample_weight=y_train['ClaimNb'])
print("Train Mean A Error:", mse)

mae = mean_absolute_error(y_test['AvgClaimAmount'], y_test_pred, sample_weight=y_test['ClaimNb'])
print("Test Mean A Error:", mae)