import imageio.v3 as iio 
from pathlib import Path
import os
from PIL import Image
import time
from e2ecv.utils import DIR_PATH, VID_SOURCE, SAVEPATH

def get_details(s: Path):
    splitted = s.stem.split('-')
    pair_id = splitted[2]
    round = splitted[-1]

    return pair_id, round

def get_vid_list(vid_path: Path):
    vid_list = [
        file
        for file in vid_path.iterdir()
        if (file.suffix.lower() == ".mp4")
        and ("dummy" not in file.stem)
    ]
    vid_list.sort()

    return vid_list

def main():
    vid_list = get_vid_list(VID_SOURCE)

    re_size_dimensions = (256, 256)
    
    resize_str = 'size-' + str(re_size_dimensions)[1:-1].replace(', ', 'x')

    start_prime = time.process_time()

    for vid in vid_list:
        p_id, r = get_details(vid)

        savepath = SAVEPATH / resize_str / p_id / f'round-{r}'
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        print(f'Path: {str(savepath)} created successfully!')

        start = time.time()

        for frame_count, frame in enumerate(iio.imiter(vid)): 
            iio.imwrite(savepath / f"{vid.stem}-frame-{frame_count}.jpg", Image.fromarray(frame).resize(re_size_dimensions)) 

        print(f'Frames from {vid.name} extracted successfully! Time taken: { time.time()- start }')

    end_prime = time.process_time()

    print(f'Total time taken: {end_prime - start_prime}')

if __name__ == '__main__':
    main()
    