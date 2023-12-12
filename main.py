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
import classify 
from recommend3 import recommend
import time
app = Flask(__name__)

# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        ################## 음악 장르 분류

        start_time = time.time()
        # 업로드 파일 처리 분기
        file = request.files['wav']

        # 파일 확장자 가져오기
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else None

        # 확장자가 .mp3나 .wav가 아닐 경우
        if file_ext not in ['mp3', 'wav']:
            return render_template('index.html', label = 'mp3 또는 wav 파일을 입력해주세요!')

        if not file: return render_template('index.html', label="No Files")
        
        input_audio_name = file.filename   #blues.00000.wav
        input_audio_path = f'input_data/audio_files/{input_audio_name}'
        input_melfiles_path=f'input_data/mel_files/{input_audio_name[:-4]}'

        if not os.path.exists(input_audio_path):
            file.save(input_audio_path) # ex) audio_files/a.wav 
            os.mkdir(input_melfiles_path) # ex) mel_files/a


        y, sr = librosa.load(input_audio_path)
        y = y[:sr*29]

        predict_genre, predict_rate = classify.genre_classify(device, file, cnn, (y, sr))
        

        play_input_audio_src, output_audio_name_list, play_output_audio_src_list = recommend(file, predict_genre,(y, sr))

        label = f'입력하신 음악의 장르는 {predict_genre} 입니다.'

        end_time = time.time()

        processing_time = end_time - start_time
        print("응답 시간: {:.2f}초".format(processing_time))
        print(play_output_audio_src_list)
        print("입력음악", file.filename)
        # 결과 리턴
        return render_template('index.html', label=label, input_audio_src = play_input_audio_src, lists = list(zip(output_audio_name_list,play_output_audio_src_list)) , predict_rate = predict_rate)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성

    device = classify.set_device()

    cnn = model.CNN().to(device)
    cnn.load_state_dict(torch.load('weight/test.pth',map_location=torch.device('cpu')))
    cnn.eval()

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=False)
