import os
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips


base_dir = 'D:/Jiwoon/dataset/hockey/'
fight_dir = os.path.join(base_dir, 'fight')
nonfight_dir = os.path.join(base_dir, 'nonfight')

def select_random_videos(folder_path, num_videos):
    videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    return random.sample(videos, min(len(videos), num_videos))


# 각 폴더에서 동영상 무작위로 150개씩 선택
selected_videos = select_random_videos(fight_dir, 150) + select_random_videos(nonfight_dir, 150)

# 선택된 동영상을 무작위 순서로 섞기
random.shuffle(selected_videos)

# 동영상 클립 합치기
clips = [VideoFileClip(video) for video in selected_videos]
final_clip = concatenate_videoclips(clips)

# 결과 동영상 저장
final_clip.write_videofile('final_video.mp4')