#!/usr/bin/python3
import glob
import os
import sys
from pathlib import Path
from multiprocessing import Pool as ProcessPool

import pandas as pd

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
from param import RAW_DATA_PATH, DATASET_PATH
from label_tools.yolov5.yolov5_helper import *

class YoloLabelTool:
    def __init__(self):
        self.rec_pixels_min = 150
        self.color_pixels_min = 30
        self.debug = False

    def process_frame(self, rgb_img_path, seg_img_path, record_name, vehicle_name):
        success = check_id(rgb_img_path, seg_img_path)
        if not success:
            return

        output_dir = f"{DATASET_PATH}/{record_name}/{vehicle_name}/yolo"
        frame_id = get_filename_from_fullpath(rgb_img_path)

        image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        image_seg = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
        if image_rgb is None or image_seg is None:
            return

        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGRA2RGB)
        img_name = os.path.basename(rgb_img_path)
        height, width, _ = image_rgb.shape

        labels_all = []
        for index, label_info in LABEL_DATAFRAME.iterrows():
            seg_color = label_info['color']
            coco_id = label_info['coco_names_index']

            mask = (image_seg == seg_color)
            tmp_mask = (mask.sum(axis=2, dtype=np.uint8) == 3)
            mono_img = np.array(tmp_mask * 255, dtype=np.uint8)

            preview_img = image_rgb
            # self.preview_img = self.image_seg
            # cv2.imshow("seg", self.preview_img)
            # cv2.imshow("mono", mono_img)
            # cv2.waitKey()
            # bordered_image = cv2.copyMakeBorder(mono_img, top=10, bottom=10, left=10, right=10, 
            #                         borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

            # contours, _ = cv2.findContours(bordered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            contours, _ = cv2.findContours(mono_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            labels = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < YoloConfig.rectangle_pixels_min:
                    continue
                # cv2.rectangle(self.preview_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # cv2.imshow("rect", self.preview_img)
                # cv2.waitKey()
                max_y, max_x, _ = image_rgb.shape
                # if y + h >= max_y or x + w >= max_x:
                #     continue

                if coco_id == TL_LIGHT_LABEL["DEFAULT"]:
                    coco_id = check_color(image_rgb[y:y + h, x:x + w, :])

                # DEBUG START
                # Draw label info to image
                cv2.putText(preview_img, COCO_NAMES[coco_id], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # DEBUG END

                label_info = "{} {} {} {} {}".format(coco_id,
                                                     float(x + (w / 2.0)) / width,
                                                     float(y + (h / 2.0)) / height,
                                                     float(w) / width,
                                                     float(h) / height)
                labels.append(label_info)
                # cv2.imshow("result", self.preview_img)
                # cv2.imshow("test", self.preview_img[y:y+h, x:x+w, :])
                # cv2.waitKey()

            if len(labels) > 0:
                labels_all += labels

        if len(labels_all) > 0:

            # print("Label output path: {}".format(self.label_out_path))
            write_image(output_dir, frame_id, image_rgb)
            write_label(output_dir, frame_id, labels_all)
        write_yaml(output_dir)
        return


if __name__ == '__main__':
    tool = YoloLabelTool()
    tool.process_frame('/home/arehman/IPAB/carla_dataset_tools/label_tools/rgb.png', '/home/arehman/IPAB/carla_dataset_tools/label_tools/s.png', 'record_name', 'vehicle_name')
