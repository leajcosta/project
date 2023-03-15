#!/usr/bin/env python3
## Doc: https://dev.intelrealsense.com/docs/python2

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import time, numpy as np
import sys, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import*
from cv_bridge import CvBridge
import signal






class camera(Node):

    def __init__(self):
        super().__init__('camera_vision')
        self.bridge=CvBridge()
        self.image_publisher = self.create_publisher(Image, '/sensor_mesgs/image', 10)
        self.depth_publisher = self.create_publisher(Image, 'depth_image', 10)
      
          
        self.infra_publisher_1 = self.create_publisher(Image, 'infrared_1',10)
        self.infra_publisher_2 = self.create_publisher(Image, 'infrared_2',10)
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.saved_img_number = 0
                

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
        pipeline, align, depth_scale, color_intrinsics = self.initialize_device()

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

    def fonctionvision(self):

    

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        print( "Connect: {device_product_line}" )
        found_rgb = True
        for s in device.sensors:
            print( "Name:" + s.get_info(rs.camera_info.name) )
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True

        if not (found_rgb):
            print("Depth camera equired !!!")
            exit(0)

        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)


        isOk=True



        # Capture ctrl-c event
        def signalInteruption(signum, frame):
            global isOk
            print( "\nCtrl-c pressed" )
            isOk= False

        signal.signal(signal.SIGINT, signalInteruption)

        count= 1
        refTime= time.process_time()
        freq= 30


        # Start streaming
        self.pipeline.start(self.config)

        while isOk:

                # Wait for a coherent tuple of frames: depth, color and accel
                frames = self.pipeline.wait_for_frames()

                depth_frame = frames.first(rs.stream.depth)
                color_frame = frames.first(rs.stream.color)

                if not (depth_frame and color_frame):
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                color_colormap = cv2.applyColorMap(cv2.convertScaleAbs(color_image, alpha=0.03), cv2.COLORMAP_JET)


                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape


                                # Load names of classes and get random colors
                classes = open('/home/bot/ros2_ws/yolodeux/yolodeux/coco.names').read().strip().split('\n')
                np.random.seed(42)
                colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

                # Give the configuration and weight files for the model and load the network.
                net = cv2.dnn.readNetFromDarknet('/home/bot/ros2_ws/yolodeux/yolodeux/yolov3.cfg', '/home/bot/ros2_ws/yolodeux/yolodeux/yolov3.weights')
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

                # determine the output layer
                ln = net.getLayerNames()
                ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
                # construct a blob from the image
                blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (416, 416), swapRB=True, crop=False)
                r = blob[0, 0, :, :]

                cv2.imshow('blob', r)
                text = f'Blob shape={blob.shape}'
                cv2.displayOverlay('blob', text)
                cv2.waitKey(1)

                net.setInput(blob)
                t0 = time.time()
                outputs = net.forward(ln)
                t = time.time()
                print('time=', t-t0)

                print(len(outputs))
                for out in outputs:
                    print(out.shape)
                def trackbar2(x):
                    confidence = x/100
                    r = r0.copy()
                    for output in np.vstack(outputs):
                        if output[4] > confidence:
                            x, y, w, h = output[:4]
                            p0 = int((x-w/2)*416), int((y-h/2)*416)
                            p1 = int((x+w/2)*416), int((y+h/2)*416)
                            cv2.rectangle(r, p0, p1, 1, 1)
                    cv2.imshow('blob', r)
                    text = f'Bbox confidence={confidence}'
                    cv2.displayOverlay('blob', text)

                r0 = blob[0, 0, :, :]
                r = r0.copy()
                cv2.imshow('blob', r)
                cv2.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
                trackbar2(50)

                boxes = []
                confidences = []
                classIDs = []
                h, w = color_image.shape[:2]

                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > 0.5:
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            box = [x, y, int(width), int(height)]
                            boxes.append(box)
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indices) > 0:
                    for i in indices.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])     
                        color = [int(c) for c in colors[classIDs[i]]]
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
                        point = (x, y)
                        print(self.position(point))
                        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i], self.position(point))
                        cv2.putText(color_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow('window', color_image)
                #cv2.destroyAllWindows()

                               # sys.stdout.write( f"\r- {color_colormap_dim} - {depth_colormap_dim} - ({round(freq)} fps)" )

                        # Show image
                images = np.hstack((color_image, depth_colormap)) # supose that depth_colormap_dim == color_colormap_dim (640x480) otherwize: resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                #images.header.stamp = node.get_clock().now().to_msg()
                         #Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
                        
                        
                #cv2.imwrite('/home/bot/Videos/image_' + str(self.saved_img_number) + '.jpg', color_image )
                self.saved_img_number += 1
                
                msg_image = self.bridge.cv2_to_imgmsg(color_image,"bgr8")
                msg_image.header.stamp = self.get_clock().now().to_msg()
                msg_image.header.frame_id = "image"
                self.image_publisher.publish(msg_image)

                # Utilisation de colormap sur l'image depth de la Realsense (image convertie en 8-bit par pixel)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(color_image, alpha=0.03), cv2.COLORMAP_JET)

                msg_depth = self.bridge.cv2_to_imgmsg(color_colormap,"bgr8")
                msg_depth.header.stamp = msg_image.header.stamp
                msg_depth.header.frame_id = "depth"
                self.depth_publisher.publish(msg_depth)
                            
     # Frequency:
                # if count == 10 :
                #     newTime= time.process_time()
                #     freq= 10/((newTime-refTime))
                #     refTime= newTime
                #     count= 0
                # count+= 1 
 

                    

                    

                            
                                    
                            
                            

                    
                                

            
        # Stop streaming
        print("\nEnding...")
        self.pipeline.stop()
                
def main(args=None):
    rclpy.init(args=args)

    Camera =camera()

    Camera.fonctionvision()
    
    # Start the ros infinit loop with the Camera node.
    rclpy.spin(Camera)

    # At the end, destroy the node explicitly.
    Camera.destroy_node()

    # and shut the light down.
    rclpy.shutdown()

if __name__ == '__main__':
    main()
