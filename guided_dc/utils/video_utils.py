import math
import os

import cv2
import imageio
import imageio.v3 as iio
import numpy as np

# def merge_rgb_array_videos(video_dir, dir_to_write):
#     # List of video file names
#     video_files = sorted(
#         [
#             f
#             for f in os.listdir(video_dir)
#             if f.endswith(".mp4") and os.path.splitext(f)[0].isdigit()
#         ],
#         key=lambda x: int(os.path.splitext(x)[0]),
#     )

#     # Load video clips
#     clips = [VideoFileClip(os.path.join(video_dir, f)) for f in video_files]

#     # Concatenate clips
#     final_clip = concatenate_videoclips(clips)

#     # Write the output to a file
#     final_clip.write_videofile(dir_to_write, codec="libx264")

#     # After merging, delete the original files
#     for file in video_files:
#         os.remove(os.path.join(video_dir, file))


def best_decomposition(num_videos):
    best_rows, best_cols = num_videos, 1  # Worst-case default
    sqrt_n = int(math.sqrt(num_videos))

    for rows in range(sqrt_n, num_videos + 1):  # Start from sqrt and go upwards
        if num_videos % rows == 0:
            cols = num_videos // rows
        else:
            cols = math.ceil(num_videos / rows)

        if cols < rows:  # Ensure cols < rows
            return rows, cols  # Return first valid pair

        best_rows, best_cols = rows, cols  # Fallback case

    return best_rows, best_cols


def merge_rgb_array_videos(input_path, output_video, num_videos=32, fps=30):
    rows, cols = best_decomposition(num_videos)

    writer = imageio.get_writer(output_video, fps=fps, quality=5)

    video_files = sorted(
        [
            f
            for f in os.listdir(input_path)
            if f.endswith(".mp4") and os.path.splitext(f)[0].isdigit()
        ],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    for i, video_file in enumerate(video_files):
        print(f"Processing video {i}")
        input_video = os.path.join(input_path, video_file)

        for frame in iio.imiter(input_video, plugin="pyav"):
            sub_frames = np.array_split(frame, num_videos, axis=0)

            # Pad with blank frames if needed
            frame_height, frame_width, channels = sub_frames[0].shape
            total_slots = rows * cols
            if len(sub_frames) < total_slots:
                blank_frame = np.zeros_like(sub_frames[0])
                sub_frames.extend([blank_frame] * (total_slots - len(sub_frames)))

            grid_frames = [
                np.hstack(sub_frames[i * cols : (i + 1) * cols]) for i in range(rows)
            ]
            stacked_frame = np.vstack(grid_frames)
            writer.append_data(stacked_frame)

    # Release video writer
    writer.close()
    print(f"Rearranged video saved as {output_video}")

    for file in video_files:
        os.remove(os.path.join(input_path, file))


def save_array_to_video(video_path, img_array, fps=30, bgr2rgb=False):
    assert (
        img_array[0].shape[2] == 3
    ), f"Image array must be in HWC order, {img_array[0].shape}"

    writer = imageio.get_writer(video_path, fps=fps, quality=5)
    for im in img_array:
        if bgr2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.append_data(im)
    writer.close()


def stack_videos_horizontally(
    video1: np.ndarray,
    video2: np.ndarray,
    output_path: str,
    fps: int = 30,
    bgr2rgb=False,
):
    """
    Stack two videos horizontally and save as a single video file.

    Args:
        video1 (np.ndarray): First video array with shape (length, width, height, channel).
        video2 (np.ndarray): Second video array with shape (length, width, height, channel).
        output_path (str): Path to save the combined video.
        fps (int): Frames per second for the output video.
    """
    # Ensure both videos have the same length
    assert (
        video1.shape[0] == video2.shape[0]
    ), "Both videos must have the same number of frames."

    # Determine the target size for resizing
    target_width = min(video1.shape[2], video2.shape[2])
    target_height = min(video1.shape[1], video2.shape[1])

    def resize_frames(frames: np.ndarray, target_size) -> np.ndarray:
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)
        return np.array(resized_frames)

    # Resize both videos
    resized_video1 = resize_frames(video1, (target_width, target_height))
    resized_video2 = resize_frames(video2, (target_width, target_height))

    # Horizontally stack the frames
    stacked_frames = [
        np.hstack((frame1, frame2))
        for frame1, frame2 in zip(resized_video1, resized_video2)
    ]

    writer = imageio.get_writer(output_path, fps=fps, quality=5)
    images_iter = stacked_frames
    for im in images_iter:
        if bgr2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.append_data(im)
    writer.close()

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # # Write frames to the output video
    # for frame in stacked_frames:
    #     out.write(
    #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     )  # Convert RGB to BGR for OpenCV

    # out.release()


def slow_down_video(input_path, output_path, slow_factor):
    """
    Slow down a .mp4 video using OpenCV.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the slowed-down video.
        slow_factor (float): Factor by which to slow down the video.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    fps = int(cap.get(cv2.CAP_PROP_FPS) / slow_factor)  # Adjust FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)  # Write frame with adjusted FPS

    cap.release()
    out.release()
    print(f"Slowed-down video saved to: {output_path}")


def accelerate_videos(video_paths, coefficient, output_paths):
    """
    Accelerate a list of videos by a given coefficient and save to new paths using OpenCV.

    Parameters:
        video_paths (list of str): List of input video file paths.
        coefficient (float): Speed-up coefficient (e.g., 2 for double speed).
        output_paths (list of str): List of output file paths where the accelerated videos will be saved.

    Returns:
        None
    """
    if len(video_paths) != len(output_paths):
        raise ValueError(
            "The number of video paths must match the number of output paths."
        )

    for input_path, output_path in zip(video_paths, output_paths):
        try:
            # Open the input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_path}")

            # Get the original video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate the new FPS
            new_fps = fps * coefficient

            writer = imageio.get_writer(output_path, fps=new_fps, quality=5)

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write every nth frame based on the coefficient
                if frame_index % int(coefficient) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame)

                frame_index += 1

            # Release resources
            cap.release()
            writer.close()

        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


if __name__ == "__main__":
    # video_paths = ["/n/fs/robot-data/guided-data-collection/data/jsg_jsg_anomaly_nominal_sim0_real250/images.mp4"]
    # output_paths = ["/n/fs/robot-data/guided-data-collection/nominal_obs.mp4"]
    # coefficient = 5
    # accelerate_videos(video_paths, coefficient, output_paths)

    merge_rgb_array_videos(
        "videos/6_grid_pick_and_place_noise/None", "videos/0.mp4", 10
    )
