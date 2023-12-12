import flask
from flask import Flask, request, render_template
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from ml import model
import shutil
import torch.nn.functional as F
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` ")
    else:
        print("GPU is enabled in this notebook.")

    return device

def scaled_logistic(x, scale=5):
    return 1 / (1 + torch.exp(-x / scale))

def genre_classify(device,file, model, audio):
    # input_wav = file.read()
        
        input_mel_path=f'input_data/mel_files/{file.filename[:-4]}'

        y, sr = audio

        segment_length = 3 * sr  # 3초 분량의 샘플 수
        for i in range(0, len(y) // (3*sr) ):
            plt.figure(figsize=(2**4, 2**4))
            plt.axis('off')
            plt.tight_layout()
            segment = y[i*segment_length:(i+1)*segment_length]
            mel_3second = librosa.feature.melspectrogram(y = segment, sr=sr)
            mel_3second_db = librosa.amplitude_to_db(mel_3second, ref=np.max)
            
            librosa.display.specshow(mel_3second_db, sr=sr)

            plt.savefig(f'{input_mel_path}/{i}.png',dpi=2**3) #bbox_inches='tight', pad_inches = 0
            plt.close()

        # 이미지를 저장할 리스트 초기화
        image_list = []

        # 폴더 안의 이미지 파일들을 반복해서 불러와 리스트에 추가
        for i in range(0, len(y) // (3*sr)):  
            image_path = f'{input_mel_path}/{i}.png'  
            image = Image.open(image_path).convert('RGB')  # 이미지 불러오기 및 RGB로 변환
            image_list.append(image)

        # 이미지를 텐서로 변환
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_list = [transform(image) for image in image_list]
        
        # 텐서를 하나로 합치기
        tensor_data = torch.stack(tensor_list)
        tensor_data = tensor_data.to(device)
        #print(tensor_data.shape) #(10,3,128,128)

        output = model(tensor_data)
        #print("output: ", output)
        _, pred = torch.max(output, 1)
        # 단계 1: Min-Max 정규화
        summed_output = torch.sum(output, dim=0)
        min_val, max_val = torch.min(summed_output), torch.max(summed_output)
        normalized_output = (summed_output - min_val) / (max_val - min_val)

        # 단계 2: 조정된 값 계산 (이번에는 제곱을 사용)
        adjusted_output = normalized_output ** 2

        # 단계 3: 재정규화 (합이 1이 되도록)
        final_output = adjusted_output / torch.sum(adjusted_output)

        # 결과를 백분율로 변환하여 출력
        percentage_output  = final_output * 100
        predict_rate = [''] * 10

        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        for i, val in enumerate(percentage_output):
            predict_rate[i] = f"{genres[i]}: {val:.2f}%"
            #print(f"Category {i}: {val:.2f}%")

        
        #print(predict_rate)

        #most_cnt = pred.mode().values.item()
        most_idx = torch.argmax(percentage_output).item()
        predict_genre =  genres[most_idx]# 모델을 통해 나온 값
        return predict_genre, predict_rate



        ''' 원래 극단적인 확률 코드
        predict_rate = [0]*10
        for i in pred:
            predict_rate[i] += 1

        print(predict_rate)

        for i in range(len(predict_rate)):
            predict_rate[i] /= 9.00
            predict_rate[i] = f"{genres[i]}: {round(predict_rate[i]*100,2)}%"
        '''