"""
Minimal prediction example
"""

import os
import csv
import random

import torch
from PIL import Image

from model.architecture import COVIDNext50
from data.transforms import val_transforms

import config

normal = []
pnemonia = []
covid = []
with open('assets/covid19newdata/Chest_xray_Corona_Metadata.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["Dataset_type"] == "TRAIN":
            row["filename"] = 'assets/covid19newdata/train/' + row["X_ray_image_name"]
            if (os.path.isfile(row["filename"])):
                if row["Label_2_Virus_category"] == "COVID-19":
                    covid.append(row)
                elif row["Label"] == "Pnemonia":
                    pnemonia.append(row)
                else:
                    normal.append(row)
random.shuffle(normal)
random.shuffle(pnemonia)
random.shuffle(covid)

# i = 0
# for img in images:
#     if img["Dataset_type"] == "TRAIN":
#         print(str(i) + ":" + img["X_ray_image_name"] + " > " + )
#         i += 1
#         if i > 100:
#             exit()

rev_mapping = {idx: name for name, idx in config.mapping.items()}

model = COVIDNext50(n_classes=len(rev_mapping))

# ckpt_pth = './experiments/ckpts/best/<model.pth>'
ckpt_pth = './experiments/COVIDNext50_NewData_F1_92.98_step_10800.pth'
# weights = torch.load(ckpt_pth)['state_dict']
weights = torch.load(ckpt_pth, map_location=torch.device('cpu'))['state_dict']
model.load_state_dict(weights)
model.eval()

transforms = val_transforms(width=config.width, height=config.height)

# for f in os.listdir('assets/covidnet/nocorona'):
    # img_pth = 'assets/covidnet/nocorona/' + f
    # img_pth = 'assets/covid_example.jpg'
for row in covid:
    img_pth = row["filename"]
    img = Image.open(img_pth).convert("RGB")
    img_tensor = transforms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        cat_id = int(torch.argmax(logits))
    print("Prediction for {} is: {}".format(img_pth, rev_mapping[cat_id]))
