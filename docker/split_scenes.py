import os
import pandas as pd
import subprocess


def split_scenes(dataset_path):

    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

    videos = os.listdir(dataset_path)

    for video in videos:

        video_path = os.path.join(dataset_path, video)
        if not os.path.isdir(video_path):
            continue

        raw_video = os.listdir(video_path)[0]
        raw_video_path = os.path.join(video_path, raw_video)

        # get meta data
        res=metadata.loc[metadata.VideoName == f'{video}','Resolution'].item()
        fps=metadata.loc[metadata.VideoName == f'{video}','FPS'].item()
        pix_fmt=metadata.loc[metadata.VideoName == f'{video}','PIX_FMT'].item()

        # split command
        split_cmd = f'scenedetect -i {raw_video_path} -o {video_path}/scenes detect-content split-video -a "-v error -f rawvideo -vcodec rawvideo -s {res} -r {fps} -pix_fmt {pix_fmt} -c:v libx264 -preset ultrafast -qp 0"'
        split_result = subprocess.run(split_cmd, capture_output=True, text=True, shell=True)
        print(split_result)
