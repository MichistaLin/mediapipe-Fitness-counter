import poseembedding as pe                      # 姿态关键点编码模块
import poseclassifier as pc                     # 姿态分类器
import extracttrainingsetkeypoints as ek        # 提取训练集关键点特征
import csv
import os


# Required structure of the images_in_folder:
#
#   fitness_poses_images_in/
#     pushups_up/
#       image_001.jpg
#       image_002.jpg
#       ...
#     pushups_down/
#       image_001.jpg
#       image_002.jpg
#       ...
#     ...

def trainset_process():
    # 如果fitness_poses_csvs_out文件夹下的squat_down.csv和squat_up.csv已经存在，则不用导入训练集再训练了
    if os.path.exists('fitness_poses_csvs_out'):
        return

    # 指定训练集的路径
    bootstrap_images_in_folder = 'fitness_poses_images_in'

    # Output folders for bootstrapped images and CSVs.
    bootstrap_images_out_folder = 'fitness_poses_images_out'
    bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

    # Initialize helper.
    bootstrap_helper = ek.BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    # Check how many pose classes and images for them are available.
    bootstrap_helper.print_images_in_statistics()

    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses were still saved in
    # the folderd (but not in the CSVs) for debug purpose. Let's remove them.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # Please manually verify predictions and remove samples (images) that has wrong pose prediction. Check as if you were asked to classify pose just from predicted landmarks. If you can't - remove it.
    # Align CSVs and image folders once you are done.

    # Align CSVs with filtered images.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

    # ## Automatic filtration
    #
    # Classify each sample against database of all other samples and check if it gets in the same class as annotated after classification.
    #
    # There can be two reasons for the outliers:
    #
    #   * **Wrong pose prediction**: In this case remove such outliers.
    #
    #   * **Wrong classification** (i.e. pose is predicted correctly and you aggree with original pose class assigned to the sample): In this case sample is from the underrepresented group (e.g. unusual angle or just very few samples). Add more similar samples and run bootstrapping from the very beginning.
    #
    # Even if you just removed some samples it makes sence to re-run automatic filtration one more time as database of poses has changed.
    #
    # **Important!!** Check that you are using the same parameters when classifying whole videos later.

    # Find outliers.

    # Transforms pose landmarks into embedding.
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Classifies give pose against database of poses.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))

    # Analyze outliers.
    bootstrap_helper.analyze_outliers(outliers)

    # Remove all outliers (if you don't want to manually pick).
    bootstrap_helper.remove_outliers(outliers)

    # Align CSVs with images after removing outliers.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()

# def dump_for_the_app():
#     pose_samples_folder = 'fitness_poses_csvs_out'
#     pose_samples_csv_path = 'fitness_poses_csvs_out.csv'
#     file_extension = 'csv'
#     file_separator = ','
#
#     # Each file in the folder represents one pose class.
#     file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]
#
#     with open(pose_samples_csv_path, 'w', newline='') as csv_out:
#         csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
#         for file_name in file_names:
#             # Use file name as pose class name.
#             class_name = file_name[:-(len(file_extension) + 1)]
#
#             # One file line: `sample_00001,x1,y1,x2,y2,....`.
#             with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
#                 csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
#                 for row in csv_in_reader:
#                     row.insert(1, class_name)
#                     csv_out_writer.writerow(row)

# dump_for_the_app()