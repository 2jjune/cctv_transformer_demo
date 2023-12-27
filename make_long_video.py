import os
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips


base_dir = 'D:/Jiwoon/dataset/hockey/'
fight_dir = os.path.join(base_dir, 'fight')
nonfight_dir = os.path.join(base_dir, 'nonfight')

# def select_random_videos(folder_path, num_videos):
#     videos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
#     return random.sample(videos, min(len(videos), num_videos))
#
#
# # 각 폴더에서 동영상 무작위로 150개씩 선택
# selected_videos = select_random_videos(nonfight_dir, 7)+select_random_videos(fight_dir, 3)
#
# # 선택된 동영상을 무작위 순서로 섞기
# # random.shuffle(selected_videos)
#
# # 동영상 클립 합치기
# clips = [VideoFileClip(video) for video in selected_videos]
# final_clip = concatenate_videoclips(clips)
#
# # 결과 동영상 저장
# final_clip.write_videofile('final_video_short.mp4')


def get_videos_with_prefix(folder_path, prefix):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(('.mp4', '.avi', '.mov'))]

# 정상 영상 선택
nonfight_videos = get_videos_with_prefix(os.path.join(base_dir, "tmp"), 'no')

# 싸우는 영상 선택
fight_videos = get_videos_with_prefix(os.path.join(base_dir, "tmp"), 'fi')

# 정상 영상 먼저 합치고, 그 다음에 싸우는 영상 합치기
clips = [VideoFileClip(video) for video in nonfight_videos + fight_videos]
final_clip = concatenate_videoclips(clips)

# 결과 동영상 저장
final_clip.write_videofile('final_video_short2.mp4')