# -*- coding: utf-8 -*-

import os
import io
import sys
import warnings
import subprocess
from sys import platform as _platform
from shutil import copyfile

gender_age_pred_repo_dir = 'age-gender-estimation'
sys.path.insert(0, gender_age_pred_repo_dir)

import numpy as np
from scipy.misc import imresize, imread
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw, ImageFont
from keras.utils.data_utils import get_file

from wide_resnet import WideResNet


FRAME_FNAME = "frame_%06d.png"

# Add Google Cloud Platform service account key to the environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_service_account_key.json"


def get_gender_age_predictor():
    weights = os.path.join(gender_age_pred_repo_dir, "pretrained_models",
                           "weights.18-4.06.hdf5")
    if not os.path.isfile(weights):
        get_file(fname="weights.18-4.06.hdf5",
                 origin="https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5",
                 file_hash='89f56a39a78454e96379348bddd78c0d',
                 cache_subdir="pretrained_models",
                 cache_dir="age-gender-estimation")

    # Initialize neural network for gender and age prediction
    gender_age_pred_im_size = 64
    gender_age_predictor = WideResNet(image_size=gender_age_pred_im_size,
                                      depth=16, k=8)()
    gender_age_predictor.load_weights(weights)
    return gender_age_predictor, gender_age_pred_im_size


def get_ffmpeg_path(ffprobe=False):
    binary = "ffprobe" if ffprobe else "ffmpeg"
    if _platform == "linux" or _platform == "linux2":  # Linux
        return os.path.join(os.sep, "usr", "bin", binary)
    elif _platform == "darwin":  # macOS (OS X)
        return os.path.join(os.sep, "opt", "local", "bin", binary)
    elif _platform == "win32":  # Windows
        return '"' + os.path.join('c:', os.sep, 'Program Files',
                                  'ffmpeg', 'bin', binary + '.exe') + '"'


def extract_frames_from_video(video_path, frames_path):
    print("Extracting frames from video...")
    subprocess.call(
        "{ffmpeg_path} -r 1 -i {video_path} -r 1 {out_path}".format(
            ffmpeg_path=get_ffmpeg_path(), video_path=video_path,
            out_path=os.path.join(frames_path, FRAME_FNAME)), shell=True)


def convert_frames_to_video(frames_path, output_video_path, fps):
    print("Creating soundless video from frames...")
    subprocess.call(
        "{ffmpeg_path} -r {frame_rate} -f image2 " 
        "-i {frames_path} -vcodec libx264 -crf {quality} -pix_fmt yuv420p "
        "{out_path}".format(
            ffmpeg_path=get_ffmpeg_path(),
            frame_rate=fps,
            frames_path=os.path.join(frames_path, FRAME_FNAME),
            quality=15,  # Lower is better
            out_path=output_video_path), shell=True)


def add_sound_from_video_to_video(sound_video_path, soundless_video_path,
                                  output_video_path):
    print("Adding sound from video to video...")
    subprocess.call(
        "{ffmpeg_path} "
        "-i {video_path_without_audio} "
        "-i {video_path_with_audio} "
        "-c copy -map 0:0 -map 1:1 -shortest {output_video_path}".format(
            ffmpeg_path=get_ffmpeg_path(),
            video_path_without_audio=soundless_video_path,
            video_path_with_audio=sound_video_path,
            output_video_path=output_video_path), shell=True)


def get_video_fps(video_path):
    result = subprocess.run(
        "{ffprobe_path} -v 0 -of csv=p=0 -select_streams v:0 "
        "-show_entries stream=r_frame_rate {video_path}".format(
            ffprobe_path=get_ffmpeg_path(ffprobe=True),
            video_path=video_path), shell=True, stdout=subprocess.PIPE)
    fps = result.stdout.decode("utf-8").strip().split('/')
    if len(fps) == 1:
        fps = float(fps[0])
    elif len(fps) == 2:
        fps = float(fps[0]) / float(fps[1])
    else:
        warnings.warn("Failed to get FPS from video! Setting it to 25.")
        fps = 25.0
    return fps


def detect_face(im_bytes):
    client = vision.ImageAnnotatorClient()
    image = types.Image(content=im_bytes)
    return client.face_detection(image=image).face_annotations


def round_to_lower_even(f):
    return np.floor(f / 2.) * 2


def detect_genders_ages(gender_age_predictor, gender_age_pred_im_size,
                        im, faces):
    genders_ages = []
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        face_im = im[box[0][1]:box[2][1], box[0][0]:box[1][0], :]
        # Crop square shaped area from the middle of the face
        square_width_height = round_to_lower_even(np.min(face_im.shape[:2]))
        half_wh = int(square_width_height / 2)
        h_center, w_center = [int(dim / 2) for dim in face_im.shape[:2]]
        face_im = face_im[h_center - half_wh: h_center + half_wh,
                          w_center - half_wh: w_center + half_wh, :]
        face_im = imresize(face_im, (gender_age_pred_im_size,
                                     gender_age_pred_im_size))
        face_im = np.expand_dims(face_im, axis=0)
        predictions = gender_age_predictor.predict(face_im)
        # Format predictions
        gender = "nainen" if predictions[0][0][0] > 0.5 else "mies"
        potential_ages = np.arange(0, 101).reshape(101, 1)
        age = int(predictions[1][0].dot(potential_ages).flatten())
        genders_ages.append([gender, age])

    return genders_ages


def highlight_faces_outlines(faces, draw, width, color):
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=width, fill=color)


def highlight_faces_parts(faces, draw, width, color):
    for face in faces:
        # Left eye
        draw.line([
            (face.landmarks[17].position.x, face.landmarks[17].position.y),
            (face.landmarks[18].position.x, face.landmarks[18].position.y),
            (face.landmarks[19].position.x, face.landmarks[19].position.y),
            (face.landmarks[16].position.x, face.landmarks[16].position.y),
            (face.landmarks[17].position.x, face.landmarks[17].position.y)],
            width=width, fill=color)
        # Right eye
        draw.line([
            (face.landmarks[21].position.x, face.landmarks[21].position.y),
            (face.landmarks[22].position.x, face.landmarks[22].position.y),
            (face.landmarks[23].position.x, face.landmarks[23].position.y),
            (face.landmarks[24].position.x, face.landmarks[24].position.y),
            (face.landmarks[21].position.x, face.landmarks[21].position.y)],
            width=width, fill=color)
        # Mouth
        draw.line([
            (face.landmarks[10].position.x, face.landmarks[10].position.y),
            (face.landmarks[8].position.x, face.landmarks[8].position.y),
            (face.landmarks[11].position.x, face.landmarks[11].position.y),
            (face.landmarks[9].position.x, face.landmarks[9].position.y),
            (face.landmarks[10].position.x, face.landmarks[10].position.y),
            (face.landmarks[12].position.x, face.landmarks[12].position.y),
            (face.landmarks[11].position.x, face.landmarks[11].position.y)],
            width=width, fill=color)


def highlight_genders_ages(faces, genders_ages, draw, color='#00ff00',
                           font_size=22):
    for face, (gender, age) in zip(faces, genders_ages):
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        try:
            font = ImageFont.truetype("Montserrat-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((box[0][0]+20, box[2][1]-120), "Ikä: %i" % age,
                  fill=color, font=font)
        draw.text((box[0][0]+20, box[2][1]-70), "Sukupuoli: %s" % gender,
                  fill=color, font=font)


def main(input_video_path, output_video_path,
         detection_start_time=None, detection_end_time=None,
         highlight_color='#00ff00', font_size=22,
         line_width_rectangle=5, line_width_face_parts=1):

    gender_age_predictor, gender_age_pred_im_size = get_gender_age_predictor()

    # Create directories for the video frames
    input_video_dir = os.path.dirname(input_video_path)
    input_frames_path = os.path.join(input_video_dir, "extracted_frames")
    output_frames_path = os.path.join(input_video_dir, "processed_frames")
    for p in [input_frames_path, output_frames_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    extract_frames_from_video(input_video_path, input_frames_path)
    fps = get_video_fps(input_video_path)

    faces, genders_ages = None, None
    for i, im_name in enumerate(os.listdir(input_frames_path)):
        print("Processing frame: %i..." % i)

        im_in_path = os.path.join(input_frames_path, im_name)
        im_out_path = os.path.join(output_frames_path, im_name)

        if ((detection_start_time and i < fps * detection_start_time) or
                (detection_end_time and i > fps * detection_end_time)):
            copyfile(im_in_path, im_out_path)
            continue

        with open(im_in_path, 'rb') as image:

            im_bytes = image.read()
            im = Image.open(io.BytesIO(im_bytes))

            if not faces or not genders_ages or i % 2 == 0:
                faces = detect_face(im_bytes)
                genders_ages = detect_genders_ages(gender_age_predictor,
                                                   gender_age_pred_im_size,
                                                   np.array(im), faces)

            draw = ImageDraw.Draw(im)

            highlight_faces_parts(faces, draw, line_width_face_parts,
                                  highlight_color)
            highlight_faces_outlines(faces, draw, line_width_rectangle,
                                     highlight_color)
            highlight_genders_ages(faces, genders_ages, draw, highlight_color,
                                   font_size)

            im.save(im_out_path)

    # Convert frames to soundless video
    _, file_ext = os.path.splitext(output_video_path)
    soundless_video_path = output_video_path.replace(file_ext,
                                                     "_soundless" + file_ext)
    convert_frames_to_video(output_frames_path, soundless_video_path, fps)

    # Add sound from the input video to the final output video
    add_sound_from_video_to_video(
        sound_video_path=input_video_path,
        soundless_video_path=soundless_video_path,
        output_video_path=output_video_path)


main(input_video_path="input.mp4",
     output_video_path="input_annotated.mp4",
     detection_start_time=2, detection_end_time=4,
     highlight_color="#00ff00", font_size=12,
     line_width_rectangle=2, line_width_face_parts=2)
