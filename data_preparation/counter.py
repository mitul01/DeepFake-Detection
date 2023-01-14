from glob import glob
from globals import BASE_PATH

video_filenames = glob(BASE_PATH + "/data/*/*.mp4")
print(len(video_filenames))
