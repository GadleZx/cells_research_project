# cells_research_project

# How to Setup

```Console
pip install -r requirements.txt
```

# How to Download Malaria Cell Images Dataset

Please download the file from the URL below.

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

# How to Use

## codes

retouch.py
retouch2.py

Process the original image (black image) so that sam2 can be used.

```Console
python retouch.py --input_dir "/home/umelab/workspace/Data/250204/VH+soil/3h/x60/Image_Sequence_1-FL" --output_dir "/home/umelab/workspace/codes/retouch_results"
```

## sam2_code

amg_pix.py

Segment all images and limit what is stored by pixel height and width size

```Console
python amg_pix.py --input_dir "/home/umelab/workspace/codes/retouch_results" --output_dir "/home/umelab/workspace/sam2/results/amg_pix" --min_height 570 --max_height 720 --min_width 570 --max_width 720
```
