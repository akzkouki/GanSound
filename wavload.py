import wave
import os
import numpy as np
import soundfile as sf
def waveloadH(filepath):
    files = os.listdir(filepath)
    sound_shape = []
    sound_data = []
    cnt = 0
    for file in enumerate(files):
        try:
            path = filepath + "\\" + file[1]
            wr = wave.open(path,"rb")
            data = wr.readframes(wr.getnframes())
            data = np.frombuffer(data,dtype="int16")
            print(str(file[1]) + " : " + str(data.shape))
            sound_data.append(data.tolist())
            sound_shape.append(data.shape[0])
            cnt += 1
        except:
            print(str(file) + "をロードできませんでした")

    print(str(cnt) +"のファイルをロードしました。")
    max = 0

    for i in sound_shape:
        if max < i:
            max = i

    sound_data_num = np.zeros([0,max])
    print(sound_data_num.shape)
    print(str(max) + "==========MAX==============")

    for i in sound_data:
        data = np.pad(i,[0,max - len(i)],"constant")
        print(data.shape)
        sound_data_num = np.vstack((sound_data_num,data))

    print(sound_data_num.shape)    
    np.save(filepath + "Data",sound_data_num) 

def waveloadL(filepath):
    files = os.listdir(filepath)
    sound_shape = []
    sound_data_list = []
    cnt = 0
    mini = 999999
    sr = 44100
    for file in enumerate(files):
        try:
            path = filepath + "\\" + file[1]
            data, samplerate = sf.read(path)

            print(samplerate)
            if samplerate == 44100:
                print(file[1])
                print(data.T.shape)
                cnt += 1
                sound_data_list.append(data.T[0])
                if mini > data.shape[0]:
                    mini = data.shape[0]


        except:
            print(str(file) + "をロードできませんでした")

    print(str(cnt) +"のファイルをロードしました。")
    print("サンプリングレートは" + str(sr) + "です")
    sound_data = np.zeros([0,mini])
    for i in sound_data_list:
        data = i[0:mini]
        sound_data = np.vstack((sound_data,data))

    print(normalize(sound_data))
    np.save(filepath + "Data",sound_data)     

def normalize(data):
    vmin = np.amin(data)
    vmax = np.amax(data)
    return (data - vmin).astype(float) / (vmax - vmin).astype(float)

waveloadL("Kick")