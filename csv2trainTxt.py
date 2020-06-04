"""
Minimal prediction example
"""

import os
import csv
import random

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

i = 0
with open('assets/covid19newdata/train_COVIDx.txt', 'w') as f:
    for counter in range(100):
        i += 1
        type = random.randint(0, 2)
        if type == 0 and len(normal) > 0:
            row = normal.pop()
            mapping = 'normal'
        elif type == 1 and len(pnemonia) > 0:
            row = pnemonia.pop()
            mapping = 'pneumonia'
        elif type == 2 and len(covid) > 0:
            row = covid.pop()
            mapping = 'COVID-19'
        else:
            continue
        f.write(str(i) + " " + row["X_ray_image_name"] + " " + mapping + "\n")
