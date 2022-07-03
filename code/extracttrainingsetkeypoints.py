import os
import sys
import csv
import cv2
import tqdm
import numpy as np
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


# 提取训练集关键点坐标
class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print('Bootstrapping ', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                # Get list of images.
                image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                # print(image_names)
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                # Bootstrap every image.
                for image_name in tqdm.tqdm(image_names):
                # for image_name in image_names:
                    # Load image.
                    input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    # Initialize fresh pose tracker and run it.
                    with mp_pose.Pose() as pose_tracker:
                        result = pose_tracker.process(image=input_frame)
                        pose_landmarks = result.pose_landmarks

                    # Save image with pose prediction (if pose was detected).
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # Save landmarks if pose was detected.
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                             for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(
                            pose_landmarks.shape)
                        # print(image_name, pose_landmarks)
                        csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str).tolist())

                    # Draw XZ projection and concatenate with the image.
                    projection_xz = self._draw_xz_projection(
                        output_frame=output_frame, pose_landmarks=pose_landmarks)
                    output_frame = np.concatenate((output_frame, projection_xz), axis=1)
            csv_out_file.close()

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None:
            return np.asarray(img)

        # Scale radius according to the image width.
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # Flip Z and move hips center to the center of the image.
            x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
            x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersetion of samples in both image folders and CSVs.
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

            # Read CSV into memory.
            rows = []
            with open(csv_out_path, newline='') as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    rows.append(row)

            # Image names left in CSV.
            image_names_in_csv = []

            # Re-write the CSV removing lines without corresponding images.
            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    # print(row)
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    if os.path.exists(image_path):
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print('Removed image from CSV: ', image_path)

            # Remove images without corresponding line in CSV.
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print('Removed image from folder: ', image_path)

    def analyze_outliers(self, outliers):
        """Classifies each sample agains all other to find outliers.

        If sample is classified differrrently than the original class - it sould
        either be deleted or more similar samples should be aadded.
        """
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """Removes outliers from the image folders."""
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder."""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder."""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([
                n for n in os.listdir(os.path.join(images_folder, pose_class_name))
                if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))
