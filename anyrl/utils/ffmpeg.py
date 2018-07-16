"""
Wrappers over the ffmpeg command.
"""

import os
import re
import subprocess

import numpy as np


def export_video(path, width, height, fps, frames, verbose=True):
    """
    Export a video file from a stream of images.

    Args:
      path: the output video path.
      width: the video width.
      height: the video height.
      fps: the frames per second.
      frames: an iterator over numpy arrays, where each
        array is a single RGB uint8 frame.
    """
    # Similar to, but simplified from, this code:
    # https://github.com/openai/retro/blob/8ffbbd148177f93e3bce85f88818fc84f9042ea5/retro/scripts/playback_movie.py#L10
    video_reader, video_writer = os.pipe()

    video_format = ['-r', str(fps), '-s', '%dx%d' % (width, height),
                    '-pix_fmt', 'rgb24', '-f', 'rawvideo']
    video_params = video_format + ['-probesize', '32', '-thread_queue_size', '10000', '-i',
                                   'pipe:%i' % video_reader]
    output_params = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-f', 'mp4',
                     '-pix_fmt', 'yuv420p', path]
    kwargs = {}
    if not verbose:
        kwargs = {'stderr': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL}
    ffmpeg_proc = subprocess.Popen(['ffmpeg', '-y', *video_params, *output_params],
                                   pass_fds=(video_reader,), stdin=subprocess.DEVNULL, **kwargs)
    try:
        for img in frames:
            assert img.shape == (height, width, 3)
            os.write(video_writer, bytes(img))
    finally:
        os.close(video_writer)
        ffmpeg_proc.wait()


def import_video(path):
    """
    Get an iterator over the frames of a video.

    Returns:
      An iterator over numpy arrays.
    """
    width, height = video_dimensions(path)
    video_reader, video_writer = os.pipe()
    try:
        args = ['ffmpeg',
                '-i', path,
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                'pipe:%i' % video_writer]
        ffmpeg_proc = subprocess.Popen(args,
                                       pass_fds=(video_writer,),
                                       stdin=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL)
        os.close(video_writer)
        video_writer = -1
        frame_size = width * height * 3
        reader = os.fdopen(video_reader, 'rb')
        while True:
            buf = reader.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype='uint8').reshape([height, width, 3])
        ffmpeg_proc.wait()
    finally:
        os.close(video_reader)
        if video_writer >= 0:
            os.close(video_writer)


def import_audio(path, sample_rate=44100):
    """
    Read audio samples from the file.

    Args:
      path: the path to a video or audio file.
      sample_rate: the number of audio samples per second.

    Returns:
      An iterator of buffers of audio samples. Each buffer
        is float array where samples are in [-1, 1].
    """
    audio_reader, audio_writer = os.pipe()
    try:
        args = ['ffmpeg',
                '-i', path,
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                'pipe:%i' % audio_writer]
        ffmpeg_proc = subprocess.Popen(args,
                                       pass_fds=(audio_writer,),
                                       stdin=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL)
        os.close(audio_writer)
        audio_writer = -1
        buffer_size = sample_rate * 2
        reader = os.fdopen(audio_reader, 'rb')
        while True:
            buf = reader.read(buffer_size)
            if not len(buf):
                break
            yield np.frombuffer(buf, dtype='int16').astype('float') / (2 ** 15)
            if len(buf) < buffer_size:
                break
        ffmpeg_proc.wait()
    finally:
        os.close(audio_reader)
        if audio_writer >= 0:
            os.close(audio_writer)


def video_dimensions(path):
    """
    Get the (width, height) of a video file.

    Args:
      path: the path to the video file.

    Returns:
      A (width, height) pair.
    """
    for line in _ffmpeg_output_lines(path):
        if not 'Video:' in line:
            continue
        match = re.search(' ([0-9]+)x([0-9]+)(,| )', line)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError('no dimensions found in output')


def video_fps(path):
    """
    Get the frame rate for a video file.

    Returns:
      A float value indicating the number of frames per
        second in the video.
    """
    for line in _ffmpeg_output_lines(path):
        if not 'Video:' in line:
            continue
        match = re.search(' ([0-9\\.]*) fps,', line)
        if match:
            return float(match.group(1))
    raise ValueError('no FPS found in output')


def _ffmpeg_output_lines(path):
    proc = subprocess.Popen(['ffmpeg', '-i', path], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    _, output = proc.communicate()
    return str(output, 'utf-8').split('\n')
