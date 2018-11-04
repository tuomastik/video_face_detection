# -*- coding: utf-8 -*-

import os
import io
import sys
import warnings
import subprocess
from sys import platform as _platform
from shutil import copyfile

import numpy as np
from scipy.misc import imresize, imread
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw, ImageFont
from keras.utils.data_utils import get_file

gender_age_pred_repo_dir = 'age-gender-estimation'
sys.path.insert(0, gender_age_pred_repo_dir)

from wide_resnet import WideResNet


class Language:
    def __init__(self):
        pass
    english = 1
    finnish = 2


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
                        im, faces, language):
    if language == Language.english:
        male, female = "male", "female"
    elif language == Language.finnish:
        male, female = "mies", "nainen"
    else:
        raise Exception("Unknown language")
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
        gender = female if predictions[0][0][0] > 0.5 else male
        potential_ages = np.arange(0, 101).reshape(101, 1)
        age = int(predictions[1][0].dot(potential_ages).flatten())
        genders_ages.append([gender, age])

    return genders_ages


def highlight_faces_outlines(faces, draw, width, color):
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=width, fill=color)


def highlight_faces_parts(faces, draw, width, color,
                          draw_eyes, draw_eyebrows, draw_mouth, draw_nose):
    for face in faces:
        if draw_eyes:
            for side in ["LEFT", "RIGHT"]:
                eye_landmarks = []
                for eye_part in ["%s_EYE_RIGHT_CORNER", "%s_EYE_BOTTOM_BOUNDARY", "%s_EYE_LEFT_CORNER",
                                 "%s_EYE_TOP_BOUNDARY", "%s_EYE_RIGHT_CORNER"]:
                    eye_part_dict = [
                        d for d in face.landmarks if
                        d.type == getattr(vision.enums.FaceAnnotation.Landmark.Type, eye_part % side)][0]
                    eye_landmarks.append((eye_part_dict.position.x, eye_part_dict.position.y))
                    draw.line(eye_landmarks, width=width, fill=color)
        if draw_eyebrows:
            for side in ["LEFT", "RIGHT"]:
                eyebrow_landmarks = []
                for eyebrow_part in ["LEFT_OF_%s_EYEBROW", "%s_EYEBROW_UPPER_MIDPOINT", "RIGHT_OF_%s_EYEBROW"]:
                    eyebrow_part_dict = [
                        d for d in face.landmarks if
                        d.type == getattr(vision.enums.FaceAnnotation.Landmark.Type, eyebrow_part % side)][0]
                    eyebrow_landmarks.append((eyebrow_part_dict.position.x, eyebrow_part_dict.position.y))
                    draw.line(eyebrow_landmarks, width=width, fill=color)
        if draw_mouth:
            mouth_landmarks = []
            for mouth_part in ["MOUTH_LEFT", "UPPER_LIP", "MOUTH_RIGHT", "LOWER_LIP",
                               "MOUTH_LEFT", "MOUTH_CENTER", "MOUTH_RIGHT"]:
                mouth_part_dict = [
                    d for d in face.landmarks if
                    d.type == getattr(vision.enums.FaceAnnotation.Landmark.Type, mouth_part)][0]
                mouth_landmarks.append((mouth_part_dict.position.x, mouth_part_dict.position.y))
                draw.line(mouth_landmarks, width=width, fill=color)
        if draw_nose:
            nose_landmarks = []
            for nose_part in ["NOSE_TIP", "NOSE_BOTTOM_RIGHT", "NOSE_BOTTOM_CENTER",
                              "NOSE_BOTTOM_LEFT", "NOSE_TIP", "MIDPOINT_BETWEEN_EYES"]:
                nose_part_dict = [
                    d for d in face.landmarks if
                    d.type == getattr(vision.enums.FaceAnnotation.Landmark.Type, nose_part)][0]
                nose_landmarks.append((nose_part_dict.position.x, nose_part_dict.position.y))
                draw.line(nose_landmarks, width=width, fill=color)


def highlight_genders_ages(faces, genders_ages, draw, color, font, font_size, language):
    try:
        font = ImageFont.truetype(font, font_size)
    except Exception:
        font = ImageFont.load_default()

    width_offset, height_offset = 20, 50

    if language == Language.english:
        age_text, gender_text = "Age", "Gender"
    elif language == Language.finnish:
        age_text, gender_text = "Ikä", "Sukupuoli"
    else:
        raise Exception("Unknown language")

    for face, (gender, age) in zip(faces, genders_ages):
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.text((box[0][0] + width_offset, box[2][1] - height_offset * 2),
                  "%s: %i" % (age_text, age), fill=color, font=font)
        draw.text((box[0][0] + width_offset, box[2][1] - height_offset),
                  "%s: %s" % (gender_text, gender), fill=color, font=font)


def main(input_video_path, output_video_path,
         detection_start_time=None, detection_end_time=None,
         highlight_color='#00ff00', font=None, font_size=22,
         line_width_rectangle=5, line_width_face_parts=1,
         language=Language.english,
         draw_eyes=True, draw_eyebrows=True, draw_mouth=True, draw_nose=True):

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
                # Make new detections every 2nd frame
                faces = detect_face(im_bytes)
                genders_ages = detect_genders_ages(gender_age_predictor, gender_age_pred_im_size,
                                                   np.array(im), faces, language)

            draw = ImageDraw.Draw(im)

            highlight_faces_parts(faces, draw, line_width_face_parts, highlight_color,
                                  draw_eyes, draw_eyebrows, draw_mouth, draw_nose)
            highlight_faces_outlines(faces, draw, line_width_rectangle, highlight_color)
            highlight_genders_ages(faces, genders_ages, draw, highlight_color, font, font_size,
                                   language)

            im.save(im_out_path)

    # Convert frames to soundless video
    _, file_ext = os.path.splitext(output_video_path)
    soundless_video_path = output_video_path.replace(file_ext, "_soundless" + file_ext)
    convert_frames_to_video(output_frames_path, soundless_video_path, fps)

    # Add sound from the input video to the final output video
    add_sound_from_video_to_video(
        sound_video_path=input_video_path,
        soundless_video_path=soundless_video_path,
        output_video_path=output_video_path)


main(input_video_path="1.mp4", output_video_path="1_annotated.mp4",
     detection_start_time=None, detection_end_time=None,
     highlight_color="#00ff00", font="Montserrat-Bold.ttf", font_size=32,
     line_width_rectangle=4, line_width_face_parts=4, language=Language.english,
     draw_eyes=True, draw_eyebrows=True, draw_mouth=True, draw_nose=True)
