import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

os.system("python3 main.py -s 360 --batch-size 12\
        --epochs 200 --lr 2e-4 --gpu mars67\
        --flag 'DeepAfterSubpixel'\
        --scale_factor 4 --n_steps 30\
        --memo 'rect\n'")
