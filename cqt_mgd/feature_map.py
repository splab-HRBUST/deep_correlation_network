# - * - coding: utf-8 - * -
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import soundfile as sf
import librosa.display
from dct_self import dct2,idct2
from scipy.signal import medfilt
from model import AttenResNet4
import torch
import scipy.io as sio  # 需要用到scipy库
import sklearn



plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font",family='SimHei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号
# bonafide_flac = "banafide.flac"
# bonafide_flac = "b_LA_T_6334757.flac"
# bonafide_flac = "b_LA_E_3494063.flac"
# bonafide_flac="b_LA_T_2319006.flac"
bonafide_flac = "b_LA_E_9999993.flac"
# bonafide_flac = "b_LA_T_3249101.flac"
# spoof_flac = "s_LA_T_6625960.flac"
# spoof_flac = "s_LA_E_5684378.flac"
# spoof_flac="s_LA_E_4852018.flac"
# spoof_flac="s_LA_E_3416290.flac"
# spoof_flac="spoof.flac"
# def displayWaveform(): # 显示语音时域波形
#     """
#     display waveform of a given speech sample
#     :param sample_name: speech sample name
#     :param fs: sample frequency
#     :return:
#     """
#     # samples, sr = librosa.load(bonafide_flac, sr=16000)
#     # samples, sr = librosa.load(spoof_flac, sr=16000)
#     # samples = samples[6000:16000]
#
#     print(len(samples), sr)
#     time = np.arange(0, len(samples)) * (1.0 / sr)
#
#     plt.plot(time, samples)
#     plt.title("语音信号时域波形")
#     plt.xlabel("时长（秒）")
#     plt.ylabel("振幅")
#     # plt.savefig("your dir\语音信号时域波形图", dpi=600)
#     plt.show()


def cqtgram(y,sr = 16000, hop_length=256, octave_bins=24,
n_octaves=8, fmin=21, perceptual_weighting=False):
    rho=0.4
    gamma=0.9
    n_xn = y*range(1,len(y)+1)
    X = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Y = librosa.cqt(
        n_xn,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    magnitude,_ = librosa.magphase(X,1)# magnitude:幅度，_:相位
    S = np.square(np.abs(magnitude)) # powerspectrum, S =   (192, 126)
    """
    medifilt中值滤波：
    中值滤波的基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，
    让周围的像素值接近真实值，从而消除孤立的噪声点
    """
    a = medfilt(S, 5) #a.shape =  (192, 251)
    dct_spec = dct2(a) # dct_spec.shape =  (192, 251)
    smooth_spec = np.abs(idct2(dct_spec[:,:291]))# smooth_spec.shape =  (192, 251)
    # smooth_spec = np.abs(a)
    gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)#对振幅的每个值都进行0.4次方处理。
    mgd = gd/(np.abs(gd)*np.power(np.abs(gd),gamma)+1e-10)
    mgd = mgd/np.max(mgd)
    cep = np.log2(np.abs(mgd)+1e-08)
    return cep
def displaySpectrum(): # 显示语音频域谱线
    x, sr = librosa.load(r'your wav file path', sr=16000)
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum，限定[:40000]
    # plt.figure(figsize=(18, 8))
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title("语音信号频域谱线")
    plt.xlabel("频率（赫兹）")
    plt.ylabel("幅度")
    plt.savefig("your dir\语音信号频谱图", dpi=600)
    plt.show()

    # # plot spectrum，不限定 [对称]
    # plt.figure(figsize=(18, 8))
    # plt.plot(frequency, magnitude)  # magnitude spectrum
    # plt.title("语音信号频域谱线")
    # plt.xlabel("频率（赫兹）")
    # plt.ylabel("幅度")
    # plt.show()





def cqtgram_true(y,sr = 16000, hop_length=256, octave_bins=24,
n_octaves=8, fmin=21, perceptual_weighting=False):
    s_complex = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
    )
    specgram = np.abs(s_complex)
    # if代码块可以不要。
    if perceptual_weighting:
       # 功率谱的感知权重：S_p[f] = frequency_weighting(f, 'A') + 10*log(S[f] / ref);
        freqs = librosa.cqt_frequencies(specgram.shape[0], fmin=fmin, bins_per_octave=octave_bins)#返回每一个cqt频率带的中心频率。
        specgram = librosa.perceptual_weighting(specgram ** 2, freqs, ref=np.max)#功率谱的感知加权。
    else:
        specgram = librosa.amplitude_to_db(specgram, ref=np.max)#将振幅谱转为用分贝表示的谱图。
    return specgram

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x
def displaySpectrogram():
    # x, sr = sf.read(spoof_flac)
    x, sr = sf.read(bonafide_flac)
    # x,sr=sf.read("b_LA_T_2319006.flac")
    z = x[192:192+192]
    print("z = ",z[48800:-1])
    # x,sr=sf.read("LA_E_5849185.flac")
    # x,sr=sf.read("s_LA_T_6625960.flac")
    y = pad(x)
    # sio.savemat("1.mat",{"waveform":y})
    # spect = cqtgram(y)
    spect = cqtgram_true(y)
    spect_npy = spect
    # print("spect = ",spect)
    model = AttenResNet4()
    save_path = "TA-networks.pth"
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_path,map_location="cpu").items()},strict=False)
    spect = torch.from_numpy(spect).unsqueeze(dim=0)
    spect = spect.to(torch.float32)
    # print(spect.dtype)
    spect1 = spect
    for i in range(7):
        spect = torch.cat([spect,spect1],dim=0)
    # print("spect.size() = ",spect.size())
    spectrogram = model(spect)
    print("spectrogram:",spectrogram.shape)
    print("=============================\n=============================")
    spectrogram = spectrogram[0]
    spectrogram = spectrogram.squeeze().squeeze()

    # print("spectrogram.size() = ",spectrogram.size())
    spectrogram = spectrogram.detach().numpy()

    # sio.savemat("spectrogram.mat",{"data":spectrogram})
    # show
    print("size - ",spectrogram.shape)
    spect_npy_norm = (spect_npy-np.mean(spect_npy))/np.std(spect_npy)
    # spectrogram = sklearn.preprocessing.scale(spectrogram, axis=0, with_mean=True, with_std=True, copy=True)
    # spectrogram = (spectrogram-np.mean(spectrogram))/np.std(spectrogram)
    print("spectrogram = ",spectrogram)
    # librosa.display.specshow(spect_npy_norm,sr=16000)
    # librosa.display.specshow(spectrogram,sr=16000)
    librosa.display.specshow(spectrogram,sr=16000)
    # librosa.display.specshow(spect)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('语音信号对数谱图')
    # plt.xlabel('时长（秒）')
    # plt.ylabel('频率（赫兹）')
    # plt.colorbar()
    plt.show()



# def displaySpectrogram1():
#     x, sr = librosa.load("spoof.flac", sr=16000)
#     print("x = ",x)
#     # compute power spectrogram with stft(short-time fourier transform):
#     # 基于stft，计算power spectrogram
#     x=pad(x)
#     spectrogram = cqtgram_true(x)
#
#     # show
#     librosa.display.specshow(spectrogram,sr=16000,y_axis="cqt_hz",x_axis="s")
#     # plt.colorbar(format='%+2.0f dB')
#     plt.title('语音信号对数谱图')
#     plt.xlabel('时长（秒）')
#     plt.ylabel('频率（赫兹）')
#     plt.show()

if __name__ == '__main__':
    # displayWaveform()
    # displaySpectrum()
    displaySpectrogram()