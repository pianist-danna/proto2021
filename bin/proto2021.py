
"""機械学習による異音検知スクリプト プロトタイプ2020年型 Ver:α0.93"""

#%%
# cording = UTF-8

"""標準ライブラリ"""
import sys
import os
import wave
import re
import random
import copy
import time
import datetime
import configparser
import json

"""サードパーティライブラリ"""
import numpy as np
import scipy
import pyaudio
import sklearn
import matplotlib.pyplot as plt
import joblib
import numba

"""個別の関数・クラスのインポート"""
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler,minmax_scale
from sklearn.decomposition import IncrementalPCA
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score,adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm

os.system("cls")    #画面をクリア
print ("Numpy version:{0}".format(np.__version__))
print ("Scipy version:{0}".format(scipy.__version__))
print ("Pyaudio version:{0}".format(pyaudio.__version__))
print ("Scikit-Learn version:{0}".format(sklearn.__version__))
print ("joblib version:{0}".format(joblib.__version__))
print ("numba version:{0}".format(numba.__version__))

"""バージョン判定 python実行環境が3.8以前であればtensorflowをインポート"""
if float(str(sys.version_info[0])+"."+str(sys.version_info[1])) >= 3.8:
    flag_tf = False
    print("Cannot use tensorflow!")
else:
    flag_tf = True
    import tensorflow as tf
    print ("Tensorflow version:{0}".format(tf.__version__))
    print ("tf.keras version:{0}".format(tf.keras.__version__))

base_dir = os.path.dirname(__file__)    #カレントフォルダを強制指定

###########################グローバル変数の初期値###########################
"""収音関連のパラメタ"""
br = 8              #ビットレート 8推奨
sr = 22050          #サンプリングレート(Hz) 20050推奨
r_wait = 5          #トリガから収音開始までのウェイト(秒) 増穂：14
r_len = 3           #収音時間(秒)   増穂：5
a_idx = None        #オーディオインデックス番号
disp_spg = False     #収音後、スペクトログラムの表示を行うかどうか Falseは波形表示

"""データセット関連のパラメタ"""
axis_freq = None        #スペクトログラムの周波数軸(リスト)
axis_tme = None         #スペクトログラムの時間軸(リスト)
aug_amt = 512           #水増し回数
file_train = "train_ds" #トレーニングデータセットのファイル名
file_test = "test_ds"   #テストデータセットのファイル名

"""モデル関連のパラメタ"""
exp_ver = 0.999                 #PCAの残存分散率
file_model_pca = "PCAmodel"     #PCAモデルのファイル名
file_model_skAE = "skAEmodel"   #scikit-learn AEモデルのファイル名

"""パス関連"""
data_dir = "data"       #データセットの保管フォルダ名 カレントの直下
save_dir = "waves"      #収音データの保管フォルダ名 dataの下
model_dir = "models"    #モデルファイルの保管フォルダ名 カレントの直下
log_dir = "logs"        #ログファイルの保管フォルダ名 カレントの直下

#############################処理系オブジェクト#############################
"""起動と初期化"""
class Init_Boot:

    def __init__(self):
        #パス関連変数の生成
        self.base_dir = base_dir
        self.data_dir = os.path.join(".\\",data_dir)
        self.save_dir = os.path.join(self.data_dir,save_dir)
        self.model_dir = os.path.join(".\\",model_dir)
        self.log_dir = os.path.join(".\\",log_dir)

        #生成するファイル名変数の生成
        self.file_train = os.path.join(
            self.data_dir,
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_" + file_train + ".npz")
        )
        self.file_test = os.path.join(
            self.data_dir,
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_" + file_test + ".npz")
        )
        self.file_model_pca = os.path.join(
            self.model_dir,
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_" + file_model_pca + ".dat")
        )

        self.file_model_skAE = os.path.join(
            self.model_dir,
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_" + file_model_skAE + ".dat")
        )

        #他の変数の初期化
        self.br = br
        self.sr = sr
        self.r_wait = r_wait
        self.r_len = r_len
        self.a_idx = a_idx
        self.disp_spg = disp_spg
        self.axis_freq = axis_freq
        self.axis_tme = axis_tme
        self.aug_amt = aug_amt
        self.exp_ver = exp_ver

        #必要なインスタンスの生成
        self.pa = pyaudio.PyAudio()
        self.cfg = configparser.ConfigParser()

    """フォルダの生成"""
    def  elem_gen_folder(self,target_dir):
        if os.path.exists(target_dir):
            pass
        else:
            os.mkdir(target_dir)
            print("Created a directory:{0}".format(target_dir))

    """設定値のセット"""
    def elem_set_ini(self):
        x = self.cfg
        x["General"] = {
            "base_dir" : self.base_dir,
            "data_dir" : self.data_dir,
            "save_dir" : self.save_dir,
            "model_dir" : self.model_dir,
            "log_dir" : self.log_dir,
            "file_train" : self.file_train,
            "file_test" : self.file_test,
            "file_model_pca" : self.file_model_pca,
            "file_model_skAE" : self.file_model_skAE
        }

        x["Rec_param"] = {
            "br" : self.br,
            "sr" : self.sr,
            "r_wait" : self.r_wait,
            "r_len" : self.r_len,
            "disp_spg" : self.disp_spg
        }

        if self.a_idx == None:
            x["Rec_param"]["a_idx"] = ""
        else:
            x["Rec_param"]["a_idx"] = str(self.a_idx)

        x["DS_param"] = {
            "aug_amt" : self.aug_amt
        }

        if self.axis_freq == None:
            x["DS_param"]["axis_freq"] = ""
        else:
            x["DS_param"]["axis_freq"] = self.axis_freq.tolist()

        if self.axis_tme == None:
            x["DS_param"]["axis_tme"] = ""
        else:
            x["DS_param"]["axis_tme"] = self.axis_tme.tolist()

        x["PCA_param"] = {
            "exp_ver" : self.exp_ver
        }

        return x

    """iniファイルへの保存"""
    def elem_save_ini(self,cfg):
        with open(
            os.path.splitext(
                os.path.basename(__file__)
            )[0] + ".ini","w") as cfgfile:
            cfg.write(cfgfile)
        print ("Saved setting parameters.")

    """iniファイルのロード"""
    def elem_load_ini(self):
        x = self.cfg
        x.read(
            os.path.join(
                "./",str(
                    os.path.splitext(
                        os.path.basename(__file__)
                    )[0] + ".ini"
                )
            )
        )
        return x

    """設定値の読み出し"""
    def elem_get_ini(self,cfg):
        x = cfg

        #読み出し
        self.base_dir = x.get("General","base_dir")
        self.data_dir = x.get("General","data_dir")
        self.save_dir = x.get("General","save_dir")
        self.model_dir = x.get("General","model_dir")
        self.log_dir = x.get("General","log_dir")
        self.file_train = x.get("General","file_train")
        self.file_test = x.get("General","file_test")
        self.file_model_pca = x.get("General","file_model_pca")
        self.file_model_skAE = x.get("General","file_model_skAE")

        self.br = x.getint("Rec_param","br")
        self.sr = x.getint("Rec_param","sr")
        self.r_wait = x.getint("Rec_param","r_wait")
        self.r_len = x.getint("Rec_param","r_len")
        self.disp_spg = x.getboolean("Rec_param","disp_spg")

        self.a_idx = x.get("Rec_param","a_idx")
        if self.a_idx == "":
            self.a_idx = None
        else:
            self.a_idx = int(self.a_idx)

        self.axis_freq = x.get("DS_param","axis_freq")
        if self.axis_freq == "":
            self.axis_freq = None
        else:
            self.axis_freq = np.array(json.loads(self.axis_freq))

        self.axis_tme = x.get("DS_param","axis_tme")
        if self.axis_tme == "":
            self.axis_tme = None
        else:
            self.axis_tme = np.array(json.loads(self.axis_tme))
        
        self.aug_amt = x.getint("DS_param","aug_amt")
        self.exp_ver = x.getfloat("PCA_param","exp_ver")

    """オーディオインデックスの定義"""
    def elem_a_idx(self):
        print("***List of available audio devices:***")
        for i in range(self.pa.get_device_count()):
            print(i,self.pa.get_device_info_by_index(i).get("name"),sep = " - ")
        x = int(input("Select Audio device Index No."))
        print("***Selected audio device #{0}.***".format(x))
        return x


    """起動処理"""
    def proc_boot(self):
        #iniファイルの有無判定
        if os.path.exists(
            os.path.join("./",str(os.path.splitext(
                os.path.basename(__file__)
                )[0] + ".ini")
            )
        ):
            #iniファイルが存在していればロードする
            cfg = self.elem_load_ini()
            self.elem_get_ini(cfg)
            print("Loaded initial settings.\n***Audio device #{0} - {1} is selected***"\
            .format(self.a_idx,self.pa.get_device_info_by_index(self.a_idx)\
                .get("name")))
            #読み出したbase_dirとグローバル変数のbase_dirが異なる場合パス関連を上書き
            if self.base_dir != base_dir:
                print("Current directory was changed. -> Overwrite config.")
                del cfg #明示的に消しておく
                self.base_dir = base_dir
                '''
                self.data_dir = os.path.join(".\\",data_dir)
                self.save_dir = os.path.join(self.data_dir,save_dir)
                self.model_dir = os.path.join(".\\",model_dir)
                self.log_dir = os.path.join(".\\",log_dir)
                self.file_train = os.path.join(
                    self.data_dir,
                    str(os.path.splitext(os.path.basename(__file__))[0]
                    + file_train + ".npz")
                )
                self.file_test = os.path.join(
                    self.data_dir,
                    str(os.path.splitext(os.path.basename(__file__))[0]
                    + file_test + ".npz")
                )
                self.file_model_pca = os.path.join(
                    self.model_dir,
                    str(os.path.splitext(os.path.basename(__file__))[0]
                    + file_model_pca + ".dat")
                )
                self.file_model_skAE = os.path.join(
                    self.model_dir,
                    str(os.path.splitext(os.path.basename(__file__))[0]
                    + file_model_skAE + ".dat")
                )
                '''
                cfg = self.elem_set_ini()
                self.elem_save_ini(cfg)
            else:
                pass

        #iniファイルが存在しない場合、オーディオインデックスを定義してセーブ
        else:
            self.a_idx = self.elem_a_idx()
            cfg = self.elem_set_ini()
            self.elem_save_ini(cfg)
        
        #データフォルダ関連の生成
        self.elem_gen_folder(self.data_dir)
        self.elem_gen_folder(self.save_dir)
        self.elem_gen_folder(os.path.join(self.save_dir,"test"))
        self.elem_gen_folder(os.path.join(self.save_dir,"test","ok"))
        self.elem_gen_folder(os.path.join(self.save_dir,"test","ng"))
        self.elem_gen_folder(os.path.join(self.save_dir,"test","valid"))
        self.elem_gen_folder(self.model_dir)
        self.elem_gen_folder(self.log_dir)

        return self.base_dir,self.data_dir,self.save_dir,self.model_dir,\
            self.log_dir,self.file_train,self.file_test,self.file_model_pca,\
            self.file_model_skAE,self.br,self.sr,self.r_wait,self.r_len,self.a_idx,\
            self.disp_spg,self.axis_freq,self.axis_tme,self.aug_amt,self.exp_ver


"""音声処理"""
class Core_Audio:
    def __init__(self):
        self.save_dir = save_dir
        self.br = br
        self.sr = sr
        self.r_wait = r_wait
        self.r_len = r_len
        self.a_idx = a_idx
        self.disp_spg = disp_spg
        self.channels = 1    #モノラル
        self.chunk = 1024   #フレームサイズ

    """収音し、バイナリデータを得る"""
    def elem_rec(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format = self.br,
            channels = self.channels,
            rate = self.sr,
            input = True,
            input_device_index = self.a_idx,
            frames_per_buffer = self.chunk
        )

        """ウェイトをかける"""
        for i in range (self.r_wait):
            if i < self.r_wait:
                print("\rWating for recording to start..." , self.r_wait-i ,end = "")
                time.sleep(1)
            print("\rWating for recording to start...0",end = "")

        """チャンクサイズごとにデータを取得し集めていく"""
        print("\rNow Recording...",end = "")
        x = []
        for i in range(0,int(sr /self.chunk * r_len)):
            x.append(stream.read(self.chunk))

        """ストリームを終了し、後処理"""
        stream.stop_stream()
        stream.close()
        pa.terminate()
        del stream

        """リスト型のデータをまとめて出力"""
        x = b"".join(x)
        print("\rNow Recording...done.")
        return x

    """wavファイルへの書き出し 入力はバイナリデータ"""
    def elem_save_wav(self,bulkwave):
        #カレントディレクトリの戻り先を取得しておく
        r_path = os.getcwd()
        os.chdir(self.save_dir)

        #ファイルネームの定義 保存時の時間
        dt = datetime.datetime.now()
        filename = dt.strftime("%Y%m%d%H%M%S") + ".wav"

        #保存する
        pa = pyaudio.PyAudio()
        x = wave.open(filename,"wb")
        x.setnchannels(1)
        x.setsampwidth(pa.get_sample_size(br))
        x.setframerate(self.sr)
        x.writeframes(bulkwave)
        x.close()
        print("Saved! Filename:{0}".format(filename))

        #カレントディレクトリを元に戻し、後処理
        os.chdir(r_path)
        del pa,x,r_path,filename

    """バイナリデータ→-1～1のNumpy配列返還"""
    def elem_BtoNP_woNorm(self,bulkwave):
        return np.frombuffer(
            bulkwave,dtype = "int" + str(self.br * 2)
        ) / float(
            (np.power(2,(self.br*2))/2)-1
        )

    """収音波形の描画"""
    def elem_vis_waveform(self,wav):
        x = np.linspace(0,len(wav)/self.sr,len(wav))
        plt.plot(x,wav) #デフォルトカラー：#1f77b4
        plt.ylim(-1,1)
        plt.ion()
        plt.show()
        plt.pause(5)
        plt.clf()
        plt.close()

    """収音スペクトログラムの描画 やや重い"""
    @numba.jit(cache = True)
    def elem_vis_spectrogram(self,wav):
        wav = wav.astype(np.float32)
        spg = np.arange(0)
        freq,time,spg = spectrogram(
            wav,
            fs = self.sr,
            window = np.hamming(self.chunk),
            nfft = self.chunk,
            scaling = "spectrum",
            mode = "magnitude"
        )
        plt.pcolormesh(
            time,freq,spg,norm = LogNorm(vmax = 1e-01,vmin = 1e-04)
        )
        plt.colorbar()
        plt.ylim(20,20000)
        plt.yscale("Log")
        plt.ion()
        plt.show()
        plt.pause(5)
        plt.clf()
        plt.close()

    """録音と保存"""
    def proc_rec_save(self):
        x = self.elem_rec() #録音する

        self.elem_save_wav(x)   #セーブ

        """録音結果の表示"""
        if self.disp_spg == False:
            self.elem_vis_waveform(self.elem_BtoNP_woNorm(x))
        else:
            self.elem_vis_spectrogram(self.elem_BtoNP_woNorm(x))


"""データセットの作成・保存・読み出し""" 
class Core_DS:
    def __init__(self):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.file_train = file_train
        self.file_test = file_test
        self.br = br
        self.sr = sr
        self.aug_amt = aug_amt
        self.chunk = 1024

    """対象フォルダ内のWaveファイルの一覧を取得"""
    def elem_wav_search(self,dir):
        x = []

        for i in os.listdir(dir):
            search_index = re.search(".wav",i)
            if search_index:
                x.append(i)
        print ("Files to process:{0}".format(x))

        return x

    """""オーディオファイルの読み込み wave_readオブジェクトを返す"""
    def elem_load_wav(self,dir,file):
        return wave.open(
            os.path.join(dir,file),"rb").readframes(-1)
    
    """バイナリデータからnumpy配列に変換しノーマライズ/スケーリング"""
    @numba.jit(cache = True)
    def elem_BtoNP_w_norm(self,wr_obj):
        return minmax_scale(
            np.frombuffer(wr_obj,dtype = "int" + str(self.br *2 )),
            feature_range=(-1,1)
        )

    """ノイズの付与"""
    @numba.jit(cache = True)
    def elem_add_noize(self,w_array):
        return w_array + np.random.randn(len(w_array))*random.uniform(0,0.01)

    """スペクトログラムの取得"""
    @numba.jit(cache = True)
    def elem_get_spg(self,noized_array):
        spg = np.arange(0)
        freq,tme,spg = spectrogram(
            noized_array.astype(np.float32),
            fs = self.sr,
            window = np.hamming(self.chunk),
            nfft = self.chunk,
            scaling = "spectrum",
            mode = "magnitude"
        )
        return freq,tme,spg

    """単一ファイルに対するAugmentation処理"""
    @numba.jit(cache = True)
    def elem_aug(self,target_ary,amount):
        for i in range(amount):
            #ノイズの付与
            wf = self.elem_add_noize(target_ary)

            #スペクトログラムの取得
            freq,tme,spg = self.elem_get_spg(wf)
            spg = spg.reshape(1,len(freq),len(tme))    #3次元配列に変換

            #水増しされたデータを積み上げる
            if i == 0:
                x= copy.deepcopy(spg)
            else:
                x= np.vstack((x,spg))
        
        return freq,tme,x

    """スペクトログラム軸のオーバーライト処理"""
    def elem_spg_axis_ini(self,freq,tme):
        ini = Init_Boot()
        x = ini.elem_load_ini()
        x["DS_param"]["axis_freq"] = str(freq.tolist())
        x["DS_param"]["axis_tme"] = str(tme.tolist())
        ini.elem_save_ini(x)
        del ini
        print("Spectrogram Axises Overwrited.")

    """データセットの作成"""
    @numba.jit
    def elem_make_ds(self,dir):
        wave_list = self.elem_wav_search(dir) #ウェーブリストを読み込む

        for i in wave_list:
            w_file = self.elem_BtoNP_w_norm(
                self.elem_load_wav(dir,i)
                )       #波形を読み込み、ノーマライズ/スケーリング

            freq,tme,auged_spg = self.elem_aug(w_file,self.aug_amt)    #Augmentation処理

            if i == wave_list[0]:
                x= copy.deepcopy(auged_spg)
            else:
                x= np.vstack((x,auged_spg))

            print("\rAugmentation count = {}".format(len(x)),end= "")

        print("\rFiles in",dir,"Augmentation done. \
            \namounts =",len(x), "\ndtype =",x.dtype)

        np.random.shuffle(x)    #混ぜる

        return freq,tme,x
    
    """X_trainの作成"""
    def proc_make_train_ds(self):
        freq,tme,x = self.elem_make_ds(self.save_dir)
        np.savez_compressed(self.file_train,X = x)
        print("The traning dataset has been generated and saved.")
        print("Dataset file :{0}".format(self.file_train))
        self.elem_spg_axis_ini(freq,tme)
        return freq,tme,x

    """X_test,y_testの作成"""
    def proc_make_test_ds(self):
        #OKデータセットの作成
        ok_dir = os.path.join(self.save_dir,"test\\ok")
        freq,tme,X_ok = self.elem_make_ds(ok_dir)
        y_ok = np.zeros(len(X_ok),dtype = "bool")   #OK:False(陽性)

        #NGデータセットの作成
        ng_dir = os.path.join(self.save_dir,"test\\ng")
        freq,tme,X_ng = self.elem_make_ds(ng_dir)
        y_ng = np.ones(len(X_ng),dtype = "bool")   #NG:True(陰性)

        #両者をスタック
        x = np.vstack((X_ok,X_ng))
        y = np.append(y_ok,y_ng)
        del ok_dir,ng_dir,X_ok,y_ok,X_ng,y_ng,freq,tme

        #ランダムシード値を生成し、両者をソート
        r_seed = np.arange(x.shape[0])
        np.random.shuffle(r_seed) 
        x = x[r_seed,:]
        y = y[r_seed,]

        #保存する
        np.savez_compressed(self.file_test,X = x,y = y)
        print("The test dataset has been generated and saved.")
        print("Dataset file :{0}".format(self.file_test))

        return x,y

    """データセットのロード"""
    def proc_load_dataset(self):
        load_0 = np.load(self.file_train)
        X_train = load_0["X"]
        del load_0
        load_1 = np.load(self.file_test)
        X_test = load_1["X"]
        y_test = load_1["y"]
        del load_1
        print ("Loaded datasets.(train/test)")

        return X_train,X_test,y_test


"""PCA分類器関連"""
class Core_PCA:
    def __init__(self):
        self.exp_ver = exp_ver
        self.scaler = MinMaxScaler()

    """エンコーダの定義とプレトレーニング"""
    @numba.jit
    def elem_train_pca(self,X):
        #1回目の処理 指定された保持分散率に基づく次元数を決める
        self.scaler.fit(X.reshape(len(X),-1))
        model = IncrementalPCA()
        model.fit(X.reshape(len(X),-1))
        n_components = np.argmax(
            np.cumsum(
                model.explained_variance_ratio_
                ) >= self.exp_ver
            ) +1

        #n_componentsが極端に少ない場合は元の次元数の1/100に制限する
        min_dim = int((X.shape[1]*X.shape[2])/100)
        if n_components < min_dim :
            n_components = min_dim
        else:
            pass

        #分散保持率と次元数をプロット
        plt.plot(np.cumsum(model.explained_variance_ratio_))
        plt.scatter(n_components,self.exp_ver)
        plt.xlabel("Dimensions")
        plt.ylabel("Explained Variance")
        plt.title(
            "Explained Ver:" + str(self.exp_ver) + \
                " -> Dimensions:" + str(n_components)
            )
        plt.show()

        #2回目処理 求めた次元数の事前学習を行い、モデルを返す
        X = self.scaler.transform(X.reshape(len(X),-1))
        x = IncrementalPCA(
            n_components = n_components
            )
        x.fit(X)

        return x,self.scaler

    """デコーダ"""
    def elem_dec_pca(self,X,model,scaler):
        x = scaler.transform(X.reshape(len(X),-1))
        x = model.inverse_transform(
            model.transform(x)
        )   #次元削減して元に戻す
        x = scaler.inverse_transform(x)
        return x.reshape(len(X),X.shape[1],X.shape[2]) #形状を元に戻す


"""オートエンコーダ関連(2020/12時点 python3.8以後では動作しない)"""
class Core_AE_keras:
    
    def __init__(self):
        self.lr = 1.0               #初期の学習率
        self.alpha = 0              #L2正則化の係数
        self.dr_rate = 0.2          #ドロップアウト率
        self.noize_rate = 0.2       #AEのノイズ付与率
        self.epochs = 100           #最大エポック数
        self.hidden_act = "selu"    #中間層の活性化関数
        self.monitor = "val_loss"   #コールバック呼び出しの指標
        self.scaler = MinMaxScaler()

    """オートエンコーダ本体の定義"""
    def def_AE(self,X):
        input_dim = (X.shape[1] * X.shape[2])
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = [X.shape[1],X.shape[2]]),
            tf.keras.layers.GaussianNoise(self.noize_rate),
            tf.keras.layers.Dense(
                int(input_dim/50),
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.alpha),
                ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(self.hidden_act),

            tf.keras.layers.Dense(
                int(input_dim/50/50),
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.alpha),
                ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(self.hidden_act),

        ])
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                int(input_dim/50),
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.alpha),
                input_shape = [int(input_dim /50/50)]
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(self.hidden_act),

            tf.keras.layers.Dense(
                input_dim,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.alpha),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("sigmoid"),
            tf.keras.layers.Reshape([X.shape[1],X.shape[2]])
        ])

        x = tf.keras.models.Sequential([encoder,decoder])
        x.compile(
            optimizer=tf.keras.optimizers.Nadam(lr = self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return x

    """早期打ち切りコールバックの定義"""
    def def_cb_earlystop(self,monitor):
        return tf.keras.callbacks.EarlyStopping(
            monitor = monitor,
            patience= 10,
            min_delta = 0.0001)

    """学習率減衰コールバックの定義"""
    def def_cb_reduce(self,monitor):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor = monitor,
            factor = 0.1,
            patience = 5,
            verbose= 1)

    """学習曲線の表示"""
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

        plt.show()

    """プレトレーニング"""
    def elem_train_AE(self,X):

        x = self.def_AE(X)

        cb_es = self.def_cb_earlystop(monitor = self.monitor)
        cb_rd = self.def_cb_reduce(monitor = self.monitor)

        X = self.scaler.fit_transform(X.reshape(len(X),-1)).reshape(len(X),X.shape[1],X.shape[2])

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

    """デコーダ"""
    def elem_dec_AE(self,X,model,scaler):
        x = scaler.transform(X.reshape(len(X),-1)).reshape(len(X),X.shape[1],X.shape[2])
        x = model.predict(x)
        return scaler.inverse_transform(x.reshape(len(X),-1)).reshape(len(X),X.shape[1],X.shape[2])
        


"""sklearnによるオートエンコーダ"""
class Core_AE_sklearn:
    def __init__(self):
        self.lr = 1.0             #初期の学習率
        self.alpha = 0              #L2正則化の係数
        self.dr_rate = 0.2          #ドロップアウト率
        self.batch_size = 32       #ミニバッチサイズ
        self.epochs = 100           #最大エポック数
        self.scaler = MinMaxScaler()

    """AEの定義"""
    def def_skl_AE(self,X):
        input_dim = (X.shape[1] * X.shape[2])
        return MLPRegressor(
            hidden_layer_sizes=(
                int(input_dim /50),
                int(input_dim/50 /50),
                int(input_dim /50)
                ),
            alpha = self.alpha,
            batch_size = self.batch_size,
            learning_rate_init = self.lr,
            max_iter = self.epochs,
            tol = 0.0001,
            verbose = True,
            early_stopping = True,
            validation_fraction = 0.05
        )

    """学習曲線の表示"""
    def elem_vis_learn_curve(self,model):
        plt.plot(model.loss_curve_)
        plt.yscale("Log")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.title("Scikit-learn Autoencoder lerning curve\n Loss = {0}".format(model.loss_))
        plt.show()

    """プレトレーニング"""
    def elem_train_skl_AE(self,X):
        x = self.def_skl_AE(X)
        print(x.get_params)
        X = self.scaler.fit_transform(X.reshape(len(X),-1))
        x.fit(X,X)

        self.elem_vis_learn_curve(x)

        return x,self.scaler

    """デコーダ"""
    def elem_dec_skl_AE(self,X,model,scaler):
        x = scaler.transform(X.reshape(len(X),-1))
        x = model.predict(x)
        x = scaler.inverse_transform(x)
        return x.reshape(len(X),X.shape[1],X.shape[2]) #形状を元に戻す


"""分類器関連、モデルデータロード"""
class Core_Estimator:
    def __init__(self):
        self.model_dir = model_dir
        self.file_model_pca = file_model_pca
        self.file_model_skAE = file_model_skAE

    """デコーダ"""  #キー入力型に改造のこと
    def elem_decord(self,X,model,scaler):
        if "IncrementalPCA" in str(model.__class__):
            dec = Core_PCA()
            X_dec = dec.elem_dec_pca(X,model,scaler)
        elif "MLPRegressor" in str(model.__class__):
            dec = Core_AE_sklearn()
            X_dec = dec.elem_dec_skl_AE(X,model,scaler)
        else:
            dec = Core_AE_keras()
            X_dec = dec.elem_dec_AE(X,model,scaler)

        return X_dec

    """元データ(X)と復元後データ(X_dec)のMSEを計算"""
    def elem_calc_mse(self,X,X_dec):
        x = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x[i] = mean_squared_error(X[i],X_dec[i])
        return x

    """分離境界の計算"""
    def elem_calc_thresh(self,y,mse):
        x = (
            (
                (mse[y].min() - mse[np.logical_not(y)].max()) / 2
                ) + mse[np.logical_not(y)].max()
            )
        print("threshold:{}".format(x))
        return x

    """MSEと分離境界を計算する
    def elem_calc_clf_thresh(self,X,X_dec,y):
        mse = self.elem_calc_mse(X,X_dec)
        if mse[y].min() > mse[np.logical_not(y)].max():
            thresh = self.elem_calc_thresh(y,mse)
        else:
            print ("Cannot define classification border threshold!!")
            thresh = None

        return mse,thresh
    """

    """k-meansクラスタリングで評価し、分離境界と分類器を生成"""
    def elem_clustering(self,X,X_dec,y):
        mse = self.elem_calc_mse(X,X_dec)

        key = False
        while key == False:
            clst = KMeans(n_clusters = 2).fit(mse[:,np.newaxis])
            if clst.cluster_centers_[0] <clst.cluster_centers_[1]:
                key = True
            else:
                key = False
        
        ami = adjusted_mutual_info_score(y,clst.labels_)
        ari = adjusted_rand_score(y,clst.labels_)
        print(
            "adjusted_mutual_info_score : {0}\nadjusted_rand_score : {1}"
            .format(ami,ari)
            )
        if ami == 1.0 and ari == 1.0:
            thresh = sum(clst.cluster_centers_) / clst.n_clusters
            clf = KNeighborsClassifier(
                n_neighbors = 2
            ).fit(mse[:,np.newaxis],y)
            print(
                "Classification border was formed.\nCluster center = {0}\nClassification border threshold = {1}"
            .format(clst.cluster_centers_,thresh))
        else:
            print ("Cannot define classification border threshold!!")
            thresh = None
            clf = None

        return mse,thresh,clst,clf

    """モデルデータと閾値のパッケージング～保存(sklearn用)"""
    def elem_save_skmodel(self,model,scaler,thresh = None,km = None,clf = None):
        x = {
            "model":model,
            "scaler":scaler,
            "thresh":thresh,
            "km" :km,
            "clf":clf
        }
        """モデルタイプの判定"""
        if "IncrementalPCA" in str(model.__class__):
            filename = self.file_model_pca
        else:
            filename = self.file_model_skAE
        joblib.dump(x,filename)
        print("Model saved : {0}".format(filename))

    """モデルデータのセーブ(keras用)"""
    def elem_save_keras(self,model,scaler,thresh = None,km = None,clf = None):
        """モデルを保存"""
        model.save(os.path.join(self.model_dir,"keras"))
        """パラメータと分類器を保存"""
        x = {
            "scaler":scaler,
            "thresh":thresh,
            "km" :km,
            "clf":clf
        }
        joblib.dump(x,os.path.join(self.model_dir,"keras",
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_keras.npz"))
        )
        print("Model saved : {0}".format(
            os.path.join(self.model_dir,"keras")
        ))

    """パッケージングファイルからモデルと閾値を取り出す(sklearn用 PCA/AE共通)"""
    def elem_load_skmodel(self,packfile):
        x = joblib.load(packfile)
        model = x["model"]
        scaler = x["scaler"]
        thresh = x["thresh"]
        km = x["km"]
        clf = x["clf"]
        if "IncrementalPCA" in str(model.__class__):
            print("Loaded learned model. Type : PCA")
        else:
            print("Loaded learned model. Type : Scikit-Learn AE")
        return model,scaler,thresh,km,clf

    """モデルと閾値の取り出し(keras用)"""
    def elem_lord_keras(self):
        model = tf.keras.models.load_model(
            os.path.join(self.model_dir,"keras")
            )
        x = joblib.load(os.path.join(self.model_dir,"keras",
            str(os.path.splitext(os.path.basename(__file__))[0]
            + "_keras.npz"))
        )
        scaler = x["scaler"]
        thresh = x["thresh"]
        km = x["km"]
        clf = x["clf"]
        print("Lorded learned model. Type : Keras denoising AE")
        return model,scaler,thresh,km,clf

    """分離境界を設定し、分類器を生成"""
    def proc_gen_estimator(self,X,y,model,scaler):
        """デコード"""
        X_dec = self.elem_decord(X,model,scaler)

        """分離境界を計算し結果を表示"""
        #mse,thresh = self.elem_calc_clf_thresh(X,X_dec,y)
        mse,thresh,km,clf = self.elem_clustering(X,X_dec,y)
        utl = Utils()
        utl.vis_estimator(y,mse,thresh,km.cluster_centers_)

        if thresh != None:
            if "keras" in str(model.__class__): #kerasモデルのセーブ
                self.elem_save_keras(model,scaler,thresh,km,clf)
            else:   #sklearnモデルのセーブ
                self.elem_save_skmodel(model,scaler,thresh,km,clf)
        else:
            pass

        return thresh,km,clf

    #モデルと閾値の取り出し keyに入力された数値で選択
    def proc_load_model(self,key):
        if key == "0": 
            model,scaler,thresh,km,clf = est.elem_load_skmodel(
                self.file_model_pca
            )
        elif key == "1":  #skAEを選択
            model,scaler,thresh,km,clf = est.elem_load_skmodel(
                self.file_model_skAE
            )
        else:
            model,scaler,thresh,km,clf = est.elem_lord_keras()

        return model,scaler,thresh,km,clf

"""推論器関連"""
class Core_Predictor:
    def __init__(self):
        #全クラスをたたき起こす
        self.rec = Core_Audio()
        self.ds = Core_DS()
        self.pca = Core_PCA()
        self.skae = Core_AE_sklearn()
        self.ae = Core_AE_keras()
        self.est = Core_Estimator()

        self.model = model
        self.scaler = scaler
        self.thresh = thresh
        self.km = km
        self.clf = clf
        self.disp_spg = disp_spg
        self.sr = sr
        self.chunk = 1024

    """確率評価と可視化"""
    @numba.jit(cache = True)
    def elem_proba_eval(self,mse_pred,y_pred,wav):
        proba =  np.count_nonzero(y_pred)/len(y_pred)
        if proba >0.5:
            result = "NG"
            col = "#ff0000"
            print("NG(True) Predict probability = {0}".format(proba))
        else:
            result = "OK"
            col = "#0000ff"
            print("OK(False) Predict probability = {0}".format(1-proba))

        plt.subplot(121)
        if self.disp_spg == False:
            x = np.linspace(0,len(wav)/self.sr,len(wav))
            plt.plot(x,wav,color = col)
            plt.ylim(-1,1)
            plt.title("Waveform")
        else:
            wav = wav.astype(np.float32)
            spg = np.arange(0)
            freq,time,spg = spectrogram(
                wav,
                fs = self.sr,
                window = np.hamming(self.chunk),
                nfft = self.chunk,
                scaling = "spectrum",
                mode = "magnitude"
            )
            plt.pcolormesh(
                time,freq,spg,norm = LogNorm(vmax = 1e-01,vmin = 1e-04)
            )
            plt.colorbar()
            plt.ylim(20,20000)
            plt.yscale("Log")
            plt.title("Spectrogram")

        plt.subplot(122)
        plt.scatter(
            np.arange(len(y_pred)),
            mse_pred,
            color = col,
            linestyle='None',
            label = "predict mse"
            )
        
        plt.plot(
            np.arange(
                len(y_pred)),
                np.full(len(y_pred),self.km.cluster_centers_[0]),
            color = "#0000ff",
            linestyle = "--",
            linewidth = 1,
            label = "False(OK) cluster center"
        )

        plt.plot(
            np.arange(
                len(y_pred)),
                np.full(len(y_pred),self.km.cluster_centers_[1]),
            color = "#ff0000",
            linestyle = "--",
            linewidth = 1,
            label = "True(NG) cluster center"
        )

        plt.plot(
            np.arange(
                len(y_pred)),
                np.full(len(y_pred),self.thresh),
                color = "#000000",
                linestyle = "--",
                label = "Threshold")

        plt.ylabel("MSE")
        plt.yscale("Log")
        plt.legend()
        plt.title(" mean MSE = {0}" .format(str(np.mean(mse_pred))))
        plt.suptitle(
            "Result : {0}".format(str(result)),
            color = col,
            size = "xx-large"
        )
        plt.ion()
        plt.show()
        plt.pause(5)
        plt.clf()
        plt.close()

        return result

    """集音～Augmentation～推論～評価"""
    @numba.jit(cache = True)
    def proc_predict(self):
        #集音し、判定用(ノーマライズあり)と表示用(ノーマライズなし)を用意
        x = self.rec.elem_rec()
        valid_wav = self.ds.elem_BtoNP_w_norm(x)
        form_wav = self.rec.elem_BtoNP_woNorm(x)

        #100回Augmentationする
        freq,tme,X_unknown,= self.ds.elem_aug(valid_wav,100)

        #流し込まれたモデルによって使用するデコーダを切り替える
        if "IncrementalPCA" in str(self.model.__class__):
            X_pred = self.pca.elem_dec_pca(X_unknown,self.model,self.scaler)
        elif "MLPRegressor" in str(self.model.__class__):
            X_pred = self.skae.elem_dec_skl_AE(X_unknown,self.model,self.scaler)
        else:
            X_pred = self.ae.elem_dec_AE(X_unknown,self.model,self.scaler)

        #mseを算出し、100回の推論を行い、確率判定
        mse_pred = self.est.elem_calc_mse(X_unknown,X_pred)
        y_pred = self.clf.predict(mse_pred[:,np.newaxis])
        result = self.elem_proba_eval(mse_pred,y_pred,form_wav)

        return valid_wav,result


#可視化ツールとデバッグコード
class Utils:
    def __init__(self):
        pass

    """分類器の可視化"""
    def vis_estimator(self,y,mse,thresh = None,cluster_centers = None):
        plt.scatter(
            y[np.logical_not(y)],
            mse[np.logical_not(y)],
            color = "#0000ff",
            linestyle='None',
            label = "False(OK) X :Cluster center"
            )
        plt.scatter(
            False,
            cluster_centers[0],
            200,
            color = "#0000ff",
            marker = "x"
        )

        plt.scatter(
            y[y],
            mse[y],
            color = "#ff0000",
            linestyle='None',
            label = "True(NG) X :Cluster center"
            )
        plt.scatter(
            True,
            cluster_centers[1],
            200,
            color = "#ff0000",
            marker = "x"
        )

        plt.plot(
            np.arange(2),
            np.array([thresh,thresh]),
            color = "#000000",
            linestyle = "--",
            label = "Threshold"
            )
        plt.xlabel("Bool_val")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.title("k-means clustering result and classification threshold")
        plt.legend()
        plt.show()


"""メニュー画面の生成"""
class SubMenus:
    def __init__(self):
        self.flag_tf = flag_tf  #tensorflowの使用可否 起動時に取得
        self.file_train = file_train
        self.model_dir = model_dir
        self.file_model_pca = file_model_pca
        self.file_model_skAE = file_model_skAE

    """モデル/データセットの有無をサーチしてリストを作る"""
    def elem_search_model(self):
        x = []

        # PCAモデルのサーチ
        if os.path.exists(self.file_model_pca) == True:
            x.append(True)
        else:
            x.append(False)

        # sk-AEモデルのサーチ
        if os.path.exists(self.file_model_skAE) == True:
            x.append(True)
        else:
            x.append(False)

        #kerasモデルのサーチ
        if os.path.exists(
            os.path.join(self.model_dir,"keras\\saved_model.pb")
        ) == True:
            x.append(True)
        else:
            x.append(False)

        #モデルデータのサーチ
        if os.path.exists(self.file_train) == True:
            x.append(True)
        else:
            x.append(False)

        return x

    """メインメニューの表示"""
    def disp_main_menu(self):
        print("\n***Please select an operating mode***")
        print("[0] : Collect recodeing data")
        print("[1] : Pre-training and define the classification border")
        print("[2] : Conduct abnormal sound judgment")
        print("[8] : Setting")
        print("[9] : Quit")


    """メニュー1生成 データセットとモデルの有無を確認しながらメニューを作る"""
    def disp_train_menu(self,status):
        x = [
            "PCA model(Light / Recommendation)",
            "Autoencoder(Medium)"
        ]

        #tensorflowが使用可能なら、AEを選択肢に入れる
        if self.flag_tf == True:
            x.append("Denoising Autoencoder(Very heavy but powerful)")
        else:
            pass

        #モデルの有無で動的にメニューの表示内容を変える        
        print("\n***Please select an train model***")
        for i in range(len(x)):
            if status[i] == True:
                x[i] = "[" + str(i) +"] : Lord/Re-train the " + x[i]
            else:
                x[i] = "[" + str(i) +"] : Train the " + x[i]

        
            print(x[i])
        print("[9] : Return to main menu")

        return np.arange(len(x)).tolist()   #有効なキーのリストを返す

    """メニュー1の分岐処理"""
    def set_flag_train(self,status,i):
        #モデルのフラグ処理 更新を要求されたらフラグを下ろす
        if status[i] == True:
            key = None
            while key == None:
                key = input(
                    "Would you like to retrain your model? [0:yes 1:no]"
                    )
                if key == "0":
                    status[i] = False
                elif key == "1":
                    pass    #何もしない…フラグはTrueのまま
                else:
                    print("\n")
                    key = None
        else:
            pass    #何もしない フラグはFalseのまま

        #データセットのフラグ処理 更新を要求されたらフラグを下ろす
        if status[3] == True:
            key = None
            while key == None:
                key = input(
                    "Do you want to rebuild the dataset? [0:yes 1:no]"
                    )
                if key == "0":
                    status[3] = False
                elif key == "1":
                    pass    #何もしない…フラグはTrueのまま
                else:
                    print("\n")
                    key = None
        else:
            pass    #何もしない フラグはFalseのまま

        return status

    """メニュー2生成"""
    def disp_pred_menu(self,status):
        x = [
            "[0] : PCA","[1] : AutoEncoder"
        ]

        key_list = []

        if self.flag_tf == True:
            x.append("[2] : Denoising Autoencoder")
        else:
            pass

        print("\n***Please select an estimator model***")
        for i in range(len(x)):
            if status[i] == True:
                print (x[i]) 
                key_list.append(i)  #有効なキーだけのリストを生成する
            else:
                pass
        print("[9] : Return to main menu")

        return key_list


#%%
#################################メイン処理#################################

if __name__ == "__main__":
  
    Boot = Init_Boot()
    base_dir,data_dir,save_dir,model_dir,log_dir,file_train,file_test,\
        file_model_pca,file_model_skAE,br,sr,r_wait,r_len,a_idx,disp_spg,\
        axis_freq,axis_tme,aug_amt,exp_ver = Boot.proc_boot()
    del Boot

    menu = SubMenus()
    model_status = menu.elem_search_model() #モデルの有無を取得

    keys = {
        "main_key" : None,
        "key_menu0" : None,
        "key_menu1" : None,
        "key_meun2" : None
    }

    while keys["main_key"] == None:
        menu.disp_main_menu() #メインメニューを表示
        keys["main_key"] = input("Select an operating mode : ")
        if keys["main_key"] =="0":    #録音モード
            rec = Core_Audio()
            while keys["key_menu0"] == None:
                keys["key_menu0"] = input(
                    "Do you want to record? [0:yes 1:no]"
                )
                if keys["key_menu0"] == "0":
                    rec.proc_rec_save()
                    keys["key_menu0"] = None
                elif keys["key_menu0"] == "1":
                    print("Exit recording mode.")
                    break
                else:
                    keys["key_menu0"] = None

            del rec
            keys["key_menu0"] = None
            keys["main_key"] = None #メインメニューに戻る

        elif keys["main_key"] == "1":   #プレトレーニングモード
            ds = Core_DS()
            while keys["key_menu1"] == None:
                valid_key1 = menu.disp_train_menu(model_status)  #サブメニューを表示し有効キーを取得
                keys["key_menu1"] = input("Select an learning model")
                if int(keys["key_menu1"]) in valid_key1:
                    menu.set_flag_train(
                        model_status,int(keys["key_menu1"])
                    )

                    if model_status[int(keys["key_menu1"])] == False:    #モデルがない場合データセットを生成
                        if model_status[3] == False:    #データセット無し/更新の場合、作る
                            print("Generating training dataset...")
                            axis_freq,axis_tme,X_train = \
                                ds.proc_make_train_ds()
                            print("Generating training dataset...")
                            X_test,y_test = ds.proc_make_test_ds()
                            model_status[3] = True      #データセットフラグを立てる

                        else:   #データセットを読み込む
                            X_train,X_test,y_test = ds.proc_load_dataset()

                        #入力されたキーに応じてプレトレーニングを行う
                        if keys["key_menu1"] == "0": 
                            pca = Core_PCA()
                            model,scaler = pca.elem_train_pca(X_train)
                            del pca

                        elif keys["key_menu1"] == "1": 
                            skae = Core_AE_sklearn()
                            model,scaler = skae.elem_train_skl_AE(X_train)
                            del skae

                        else: 
                            krae = Core_AE_keras()
                            model,scaler = krae.elem_train_AE(X_train)
                            del krae

                        est = Core_Estimator()
                        thresh,km,clf = est.proc_gen_estimator(
                            X_test,y_test,model,scaler
                        )
                        if thresh != None:
                            model_status[int(keys["key_menu1"])] = True   #作成したモデルのフラグを立てる
                        del est

                    else:
                        est = Core_Estimator()
                        #入力されたキーに応じてモデルを読み込む
                        model,scaler,thresh,km,clf = est.proc_load_model(keys["key_menu1"])
                        del est
                    break   #要らなさそうだけどお呪い

                elif keys["key_menu1"] == "9":  #ループを抜ける
                    break
                else:
                    keys["key_menu1"] = None
            del ds,valid_key1
            keys["key_menu1"] = None
            keys["main_key"] = None #ループを抜けたらメインメニューに戻る

        elif keys["main_key"] == "2":   #推論モード
            if model_status[0] == False \
                and model_status[1] == False \
                and model_status[2] == False :  #モデルが一つもなければ回れ右
                print ("Cannot predict because there is no model.\nReturn to the main menu")
                keys["main_key"] = None
            else:
                while keys["key_meun2"] == None:
                    valid_key2 = menu.disp_pred_menu(model_status) #サブメニューを表示し有効キーを取得
                    keys["key_menu2"] = input("Select an learning model")
                    if int(keys["key_menu2"]) in valid_key2:
                        #モデルの読み込み
                        try:
                            model
                        except: #モデルが無い場合、入力されたキーに応じて読み込む
                            est = Core_Estimator()
                            model,scaler,thresh,km,clf = est.proc_load_model(keys["key_menu2"])
                            del est
                        else:
                            pass
                        
                        pred = Core_Predictor()
                        pred.proc_predict()
                        del pred

                    elif keys["key_menu2"] == "9":  #ループを抜ける
                        break
                    else:
                        keys["key_menu2"] = None
                del valid_key2
                keys["key_menu2"] = None
                keys["main_key"] = None #ループを抜けたらメインメニューに戻る

        elif keys["main_key"] == "8":   #セッティングモード 工事中
            print("\nSorry,Cullenly under construction.\n")
            keys["main_key"] = None

        elif keys["main_key"] == "9":   #スクリプトの終了
            print("\nQuit.\n")
            break

        else:    #定義されてないキーの処理
            print("\n")
            keys["main_key"] = None
        
    keys["main_key"] = None
    del menu
    print("スクリプトは正常終了しました")
    

# %% [markdown]
'''
以下備忘メモ

やること
- メイン引数の辞書化
- セッティングモードの実装
- 偽判定処理 …直前の推論処理の論理とデータをメイン変数に残し、渡す

20210104 v0.93
    - コード全面書換え
        - クラス構造を整理 旧定義系と旧処理系をまとめ、関数名のサフィックスで区別
        - 継承の全面廃止 コンストラクタと委譲で名前空間を明確化
        - UI関連のクラスを表示とデータ収集に限定し、処理はメイン処理に移動
        - UI関連のキー入力状態をリストに格納させ、UIの制御構文を簡素化
        - モデル及びデータセットの有無をフラグ管理しUIを動的に変化させる
        - 各種進捗表示を充実
        - リファクタリング(主に高水準APIの導入と構文の簡素化,例外処理の廃止)
    - numbaを導入し一部処理を高速化
    - 起動時に実行環境のバージョン判定(tensorflow動かない対策)
        - Anacondaでtensorflowが動かない環境の場合、tensorflow関連の機能をロック
    - カレントディレクトリ以外のパス指定を相対パス化(ネットワークブート対策)
    - 収音ウェイト時の残り時間/収音中/収音終了が表示されるように
    - 収音後の波形/スペクトログラム表示の横軸を時間軸に
    - Augmentationの進捗表示を変更 改行をなくし、終了時に情報を提示
    - Augmentation実施時、もしくはデータセットロード時に  
        スペクトログラムの周波数/時間情報を読み込むように  
        →スペクトログラム表示時、周波数軸/時間軸を表示
    - スペクトログラムのデータ量を半分に(float64 →float32)
    - スケーラーを導入 主にオートエンコーダ系の誤差縮小対策
    - `sklearn`の`MLPRegressor`を使用したオートエンコーダを実装 →将来的に廃止の予定
    - `tf.keras`のノイズ除去オートエンコーダを実装 →リカレント型に改良予定
    - Estimatorを変更 手計算をやめ、k近傍法による機械学習判定に変更
        - k平均法クラスタリングのARI/AMIが100(y_testと同一の結果)かどうかを確認し、
            同一である(k近傍法で同様の結果が得られる)場合にk近傍法分類器を設定
            閾値はクラスタセンタ間中央値に取る(目安以上の意味はなくなっている)
    - 確率判定画面表示を整理 タイトルで判定結果を大書き+色分け
    - 確率判定結果表示時、予測確率を表示するように
    - 確率判定結果のグラフでクラスタセンタを表示
    - モデル/データセットの有無でプレトレーニングモードの表示を動的に変更するように
    - モデルの有無で、推論モードで選択できるモデルタイプが制限されるように

20201205 ｖ0.92
    入力信号のレベルが低い際に特徴量が埋没しないよう、
        データセット生成前に波形をノーマライズするように
    推論結果で波形を表示するように
    推論結果の波形とMSEを、判定結果により色が変わるように(判りにくいので)
    録音後や推論後の波形表示時、レンジを正しく(レベルが判るように)表示するように
    生成されるiniファイルの名前が自分自身の名前に連動するように(現場対応)
    起動時、モデルファイルがあれば自動読み込みするように(現場対応)
    怨念のように残っていた古いコメントコードを掃除

20201119　v0.91
    録音トリガーオン後、r_wait_lengthの時間分待機するように
    録音終了後、5秒間波形を表示し、セーブ処理に移行するように
    メインメニューのキー判定を未定義キー優先に(ばかよけ対応)

20200921 v0.90
    プレトレーニング及び確率判定を活性化

20200816
    Rec.pyおよびAE.py(PCAのみ)の機能取り込み・結合
    簡易的CUIユーザーインターフェース実装
    現時点では録音機能のみ活性化 他はデバッグモードで呼び出し

'''