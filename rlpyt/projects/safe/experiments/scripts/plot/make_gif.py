import os
import sys
import imageio
import re
from tqdm import tqdm
import shutil

def main(png_dir, delete_pngs):
    images = []
    scenarioname = ""
    scenarioregex = r'(.*)_ts_(\d+)\.png'
    files = []
    for file_name in tqdm(sorted(os.listdir(png_dir)), desc='Making gif'):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            files.append(file_path)
            images.append(imageio.imread(file_path))
            scenarioname = re.match(scenarioregex, file_name).group(1)
    imageio.mimsave(os.path.join(png_dir, f'{scenarioname}.gif'), images)

    if delete_pngs:
        for file in files:
            os.remove(file)


if __name__ == '__main__':
    main(*sys.argv[1:])
