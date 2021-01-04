# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## やりたいこと
# `proto2021.py`のリカレントオートエンコーダ化検討

# %%
import os

os.chdir("C:\\Users\\amane\\OneDrive\\python\\proto2021\\bin")
print(os.getcwd())


# %%
import numpy as np
X = np.load(".\\data\\proto2021_train_ds.npz")
X_train = X["X"]
del X


# %%
X_train.shape

# %% [markdown]
# ---
# 4層のノイズ除去リカレントオートエンコーダ 入出力段で行列を転置し、スペクトログラムの横軸(列方向)がタイムステップになるように調整

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def def_early_stopping():
    return tf.keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience= 10,
            min_delta = 0.0001)

def def_reduce():
    return tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            patience = 5,
            verbose= 1)

encoder = tf.keras.models.Sequential([
    tf.keras.layers.Permute((2,1),input_shape=[X_train.shape[1], X_train.shape[2]]),
    tf.keras.layers.GaussianNoise(0.2),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2), return_sequences=True),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2 /2), return_sequences=True),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2 /2 / 2), return_sequences=True),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2 / 2 / 2 / 2))
],name = "encoder")
decoder = tf.keras.models.Sequential([
    tf.keras.layers.RepeatVector(X_train.shape[2], input_shape=[int(X_train.shape[1] / 2 / 2 / 2 / 2)]),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2 / 2 / 2), return_sequences=True),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2 / 2), return_sequences=True),
    tf.keras.layers.GRU(int(X_train.shape[1] / 2), return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X_train.shape[1], activation="sigmoid")),
    tf.keras.layers.Permute((2,1))
],name = "decoder")
rec_ae = tf.keras.models.Sequential([encoder, decoder],name = "autoencoder")
rec_ae.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Nadam(),
                     metrics=["accuracy"])

cb_es = def_early_stopping()
cb_rd = def_reduce()
scaler = MinMaxScaler()

rec_ae.summary()

X_train = scaler.fit_transform(X_train.reshape(len(X_train),-1)).reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])
hist = rec_ae.fit(X_train, X_train, epochs=1000, validation_split = 0.05,callbacks=[cb_es,cb_rd],shuffle = True,use_multiprocessing=True)
X_train = scaler.inverse_transform(X_train.reshape(len(X_train),-1)).reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])

plt.subplot(121)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.yscale("log")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(["Train","val"])
plt.title("Accuracy")

plt.subplot(122)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["Train","val"])
plt.title("Loss")

plt.suptitle("Learning curve")
plt.show()

# %% [markdown]
# セル数が周波数方向(513次元)しかないので、驚異的な速度で学習が進む
# %% [markdown]
# 移植に向けてクラス化

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Core_AE_keras:
    def __init__(self):
        self.epochs = 1000
        self.reduce = 2             #次元数の減衰率の逆数(2だと、1段ごとに1/2になる)
        self.noize_rate = 0.2       #AEのノイズ付与率 0.2推奨
        self.monitor = "val_loss"   #コールバック呼び出しの指標
        self.scaler = MinMaxScaler()

    def def_rAE(self,X):
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Permute((2,1),input_shape=[X.shape[1], X.shape[2]]),
            tf.keras.layers.GaussianNoise(self.noize_rate),
            tf.keras.layers.GRU(int(X.shape[1] / self.reduce), return_sequences=True),
            tf.keras.layers.GRU(int(X.shape[1] / (self.reduce ** 2)), return_sequences=True),
            tf.keras.layers.GRU(int(X.shape[1] / (self.reduce ** 3)), return_sequences=True),
            tf.keras.layers.GRU(int(X.shape[1] / (self.reduce ** 4)))
            ],name = "encoder")
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.RepeatVector(X.shape[2], input_shape=[int(X.shape[1] / (self.reduce ** 4))]),
            tf.keras.layers.GRU(int(X.shape[1] / (self.reduce ** 3)), return_sequences=True),
            tf.keras.layers.GRU(int(X.shape[1] / (self.reduce ** 2)), return_sequences=True),
            tf.keras.layers.GRU(int(X.shape[1] / self.reduce), return_sequences=True),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(X.shape[1], activation="sigmoid")
            ),
            tf.keras.layers.Permute((2,1))
            ],name = "decoder")
        x = tf.keras.models.Sequential([encoder, decoder],name = "autoencoder")
        x.compile(
            loss="binary_crossentropy", 
            optimizer=tf.keras.optimizers.Nadam(),
            metrics=["accuracy"]
        )

        return x

    def def_early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor = self.monitor,
            patience= 10,
            min_delta = 0.0001)

    def def_reduce(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor = self.monitor,
            patience = 5,
            verbose= 1)

    def elem_vis_learn_curve(self,hist):
        plt.subplot(121)
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.legend(["Train","val"])
        plt.title("Accuracy")

        plt.subplot(122)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend(["Train","val"])
        plt.title("Loss")

        plt.suptitle("Learning curve")
        plt.show()

    def elem_train_AE(self,X):
        x = self.def_rAE(X)
        
        cb_es = self.def_early_stopping()
        cb_rd = self.def_reduce()
        X = self.scaler.fit_transform(
            X.reshape(len(X),-1)
        ).reshape(len(X),X.shape[1],X.shape[2])

        x.summary()

        hist = x.fit(
            X,
            X,
            epochs = self.epochs,
            callbacks = [cb_es,cb_rd],
            validation_split = 0.05,
            shuffle = True,
            use_multiprocessing=True
        )

        print('Autoencoder learning is over!')
        x.summary()
        self.elem_vis_learn_curve(hist)

        return x,self.scaler

ae = Core_AE_keras()
rec_ae,scaler = ae.elem_train_AE(X_train)
X_train = scaler.inverse_transform(X_train.reshape(len(X_train),-1)).reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])


# %%
X = np.load(".\\data\\proto2021_test_ds.npz")
X_test = X["X"]
y_test = X["y"]
del X

X_test = scaler.transform(X_test.reshape(len(X_test),-1)).reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2])
X_dec = rec_ae.predict(X_test)
X_test = scaler.inverse_transform(X_test.reshape(len(X_test),-1)).reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2])
X_dec = scaler.inverse_transform(X_dec.reshape(len(X_dec),-1)).reshape(X_dec.shape[0],X_dec.shape[1],X_dec.shape[2])


# %%
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error
mse = np.zeros(X_test.shape[0])
for i in range(len(X_test)):
    mse[i] = mean_squared_error(X_test[i],X_dec[i])


for i in range(5):
    plt.subplot(2,1,1)
    plt.pcolormesh(X_test[i],norm = LogNorm(vmax = 1e-01,vmin = 1e-04))
    plt.colorbar()
    plt.title("X_test[{0}],label = {1}".format(i,str(y_test[i])))
    plt.subplot(2,1,2)
    plt.pcolormesh(X_dec[i],norm = LogNorm(vmax = 1e-01,vmin = 1e-04))
    plt.colorbar()
    plt.title("X_dec[{0}],mse = {1}".format(i,mse[i]))
    plt.suptitle("Decording error")
    plt.savefig("image.png")
    plt.close()

    img = plt.imread("image.png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# %%
pred = np.vstack((y_test.T,mse.T)).T
for i in range(20):
    print(pred[i])

# %% [markdown]
# ---
# 
# ## わかったこと
# - 時間軸方向をリカレント化することで、計算負荷を大幅に減らせる
# - 時間軸方向の特徴はちゃんとつかんでおり、従来問題だった時間軸方向の弱さをカバーできている
# 
# ## 次にやること
# 移植

# %%



