import cv2
import numpy as np
import pyrealsense2 as rs


def initialize_device():
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()


    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get stream profile and camera intrinsics
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    # print(color_intrinsics)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)



    return pipeline, align, depth_scale,color_intrinsics


def position(point=(0,0)):
    point = (x, y)
    # Create REALSENSE  pipeline
    pipeline, align, depth_scale, color_intrinsics = initialize_device()

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    # if not aligned_depth_frame or not color_frame:
    #     continue

    depth_map = np.asanyarray(aligned_depth_frame.get_data()) * depth_scale * 1000
    dist = aligned_depth_frame.get_distance(int(point[0]),
                                            int(point[1])) * 1000  # convert to mm
    # calculate real RGB world coordinates
    X = dist * (point[0] - color_intrinsics.ppx) / color_intrinsics.fx
    Y = dist * (point[1] - color_intrinsics.ppy) / color_intrinsics.fy
    Z = dist

    # calculate  real center of the realsense world coordinates in mm
    X = X - 35
    Y = Y
    Z = Z

    # calculate  real center of the realsense world coordinates in meter
    X = X/1000
    Y = Y/1000
    Z = Z/1000
    position=(X,Y,Z)


    return position


if __name__ == '__main__':

    x = 0
    y = 0
    point = (x, y)
    print("position(point)")
    print(position(point))
