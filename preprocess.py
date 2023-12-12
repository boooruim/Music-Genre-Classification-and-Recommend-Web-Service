
import librosa
import numpy as np
import os
if __name__ == '__main__':

    genre_name = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']


    parent_directory = 'input_data'
    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)

    folder_path = os.path.join(parent_directory, 'audio_files')
    os.mkdir(folder_path)
    
    folder_path = os.path.join(parent_directory, 'mel_files')
    os.mkdir(folder_path)


    parent_directory = 'Data'

    if not os.path.exists(parent_directory):
            os.mkdir(parent_directory)


    for i in range(10):
        folder_path = os.path.join(parent_directory, genre_name[i])
    
        # 폴더가 이미 존재하지 않는 경우에만 생성
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    original_data_dir = '../archive/genres_original/'

    for i in range(10):

        for j in range(100):

            original_data = original_data_dir+genre_name[i]+f"/{genre_name[i]}.{j:05d}.wav"

            if not os.path.exists(original_data):
                continue

            y, sr = librosa.load(original_data)
            y = y[:sr*29]

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            np.save(f'{parent_directory}/{genre_name[i]}/{j:05d}.npy',chroma)
