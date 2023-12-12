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
import time
from scipy.spatial.distance import cosine, euclidean
from sklearn.neighbors import NearestNeighbors


def recommend(file, predict_genre, audio):

    y, sr = audio

    input_chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    #print(input_chroma.shape)  

    # 폴더에 있는 100개의 Chroma 파일 로드
    chroma_folder_path = f"Data/{predict_genre}"
    
    chromas = []
    file_names = []

        
    # 각 장르 폴더 내의 모든 파일을 순회합니다.
    for filename in os.listdir(chroma_folder_path):
        if file.filename[:-4] != filename[:-4] :
            file_path = os.path.join(chroma_folder_path, filename)
            
            # NumPy 파일로부터 크로마 데이터를 불러옵니다.
            chroma = np.load(file_path)
            chromas.append(chroma.flatten())
            file_names.append(filename)


    # NearestNeighbors 모델을 초기화하고, 데이터로 학습시킵니다.
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(chromas)


    # 가장 가까운 이웃을 찾습니다.
    distances, indices = nn.kneighbors(input_chroma.flatten().reshape(1, -1))
    #print(indices)
    # 가장 가까운 크로마와 해당 장르를 출력합니다.
    print(distances)
    print(indices)


    source_directory = f'../archive/genres_original/{predict_genre}'
    destination_directory = "./static"

    #output_audio_name = os.listdir(source_directory)[most_similar_idx]
    

    for filename in os.listdir(destination_directory):
        # .mp3 파일을 찾는다
        if filename.endswith('.mp3'):
            os.remove(destination_directory+'/'+ filename)


    input_audio_name = file.filename
    input_source_path = os.path.join(f'./input_data/audio_files', input_audio_name)
    input_play_audio_src = f'input_{str(time.time())}.mp3'
    destination_path = os.path.join(destination_directory, input_play_audio_src)  
    shutil.copy2(input_source_path, destination_path)

    output_play_audio_src_list = []
    output_audio_name_list = []
    for i in (indices[0]): # [[81 55 28]]
        closest_name = file_names[i]

        output_audio_name = closest_name[:-4]+'.wav'
        output_audio_name_list.append(output_audio_name)

        output_source_path = os.path.join(source_directory, output_audio_name)
        output_play_audio_src = f'output_{str(time.time())}.mp3'
        output_play_audio_src_list.append(output_play_audio_src)

        destination_path = os.path.join(destination_directory, output_play_audio_src)  # 대상 파일 이름을 'temp.mp3'로 설정
        shutil.copy2(output_source_path, destination_path)

    

    return (input_play_audio_src, output_audio_name_list,  output_play_audio_src_list)