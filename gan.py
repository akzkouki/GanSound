import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

from keras.optimizers import SGD
from keras.datasets import mnist
from keras.regularizers import l1_l2

import soundfile as sf
import shutil
from matplotlib import animation

def init(filepath):
    data = np.load(filepath)
    
    input_dim = data.shape[1]

    print("InputShape=" + str(input_dim) + "Type" + str(type(input_dim)))
    wav_write("test",data[0])
    return input_dim,data

def wav_write(filename,data):
    samplerate = 44100
    filepath = "GanSound"
    try:
        os.mkdir("./" + filepath)
    except:
        pass
    # print(filepath + "の中身を空にしてから実行してください")
    sf.write(filename + ".wav", data, samplerate)
    shutil.move("./" + filename + ".wav", "./" + filepath)

def generator():
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)
    Gen = Sequential()
    Gen.add(Dense(input_dim=100, units=256, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Gen.add(BatchNormalization(mode=0))
    Gen.add(act)
    Gen.add(Dense(units=512, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Gen.add(BatchNormalization(mode=0))
    Gen.add(act)
    Gen.add(Dense(units=1024, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Gen.add(BatchNormalization(mode=0))
    Gen.add(act)
    Gen.add(Dense(units=input_dim, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Gen.add(BatchNormalization(mode=0))
    # Gen.add(Activation("sigmoid"))
    Gen.add(Activation("tanh"))
    generator_optimizer = SGD(lr=0.1, momentum=0.3, decay=1e-5)
    # Gen.compile(loss="binary_crossentropy", optimizer=generator_optimizer)
    return Gen

def discriminator():
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)
    Dis = Sequential()
    Dis.add(Dense(input_dim=input_dim, units=1024, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Dis.add(act)
    Dis.add(Dense(units=512, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Dis.add(act)
    Dis.add(Dense(units=256, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Dis.add(act)
    Dis.add(Dense(units=1, kernel_regularizer=l1_l2(1e-5, 1e-5)))
    Dis.add(Activation("sigmoid"))
    discriminator_optimizer = SGD(lr=0.1, momentum=0.1, decay=1e-5)
    Dis.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)
    return Dis

def generative_adversarial_network(generator_model, discriminator_model):
    GAN = Sequential()
    GAN.add(generator_model)
    discriminator_model.trainable=False
    GAN.add(discriminator_model)
    gan_optimizer = SGD(0.1, momentum=0.3)
    GAN.compile(loss="binary_crossentropy", optimizer=gan_optimizer)
    return GAN

def plot_metrics(metrics, epoch=None):
    plt.figure(figsize=(10,8))
    plt.plot(metrics["d"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join("metrics", "dloss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["g"], label="generative loss", color="r")
    plt.legend()
    plt.savefig(os.path.join("metrics", "g_loss" + str(epoch) + ".png"))
    plt.close()
    print("PlotOut!!")


def main_train(z_input_size, generator_model, discriminator_model, gan_model, loss_dict, X_train, generated_figures=None, z_group=None, z_plot_freq=200, epoch=1000, plot_freq=25, batch=100):
    write_rate = 50

    with tqdm(total=epoch) as pbar:
        for e in range(epoch):
            pbar.update(1)
            noise = np.random.uniform(0, 1, size=[batch, z_input_size])

            generated_images = generator_model.predict_on_batch(noise)

            print(np.amax(generated_images))
            print(np.amin(generated_images))
            X = np.vstack((X_train, generated_images))

            # ラベル作成
            y = np.zeros(int(data.shape[0] + batch))
            y[batch:] = 1
            y = y.astype(int)

            # discriminatorの学習
            discriminator_model.trainable = True
            d_loss = discriminator_model.train_on_batch(x=X, y=y)
            discriminator_model.trainable = False

            # generatorの学習
            noise = np.random.uniform(0, 1, size=[batch, z_input_size])
            y = np.zeros(batch)
            y = y.astype(int)
            g_loss = gan_model.train_on_batch(x=noise, y=y)

            
            loss_dict["d"].append(d_loss)
            loss_dict["g"].append(g_loss)
            if e == epoch-1:
                plot_metrics(loss_dict, int(e/plot_freq))

            if(e % write_rate == 0):
                wav_write("epoch=" + str(e) ,generated_images[0])


input_dim , data = init("KickData.npy")

X_train = data
X_train.astype('float32')

print(data[0])

print("X_train shape", X_train.shape)
print(X_train.shape[0], "train samples")


gen = generator()
dis = discriminator()
gan = generative_adversarial_network(gen, dis)
gan.summary()

gen.summary()
dis.summary()
# パラメータ設定
gan_losses = {"d":[], "g":[], "f":[]}
epoch = 50
batch = 1000
z_plot_freq = 1000
plot_freq = epoch -2
z_input_vector = 100
n_train_samples = 60000
examples = 9

z_group_matrix = np.random.uniform(0,1,examples*z_input_vector)
z_group_matrix = z_group_matrix.reshape([9, z_input_vector])
print(z_group_matrix.shape)

generated_figures = []
main_train(100, gen, dis, gan, loss_dict=gan_losses, X_train=X_train, generated_figures=input_dim, z_group=z_group_matrix, z_plot_freq=z_plot_freq, epoch=epoch, plot_freq=plot_freq, batch=batch)

  
     

    