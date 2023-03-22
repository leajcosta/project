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

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)

        # Get stream profile and camera intrinsics
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.color_intrinsics = color_profile.get_intrinsics()
        # print(color_intrinsics)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
                

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
       # self.pipeline.start(self.config)

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
                classes = open('/home/bot/ros2_ws/yolodeux/coco.names').read().strip().split('\n')
                np.random.seed(42)
                colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

                # Give the configuration and weight files for the model and load the network.
                net = cv2.dnn.readNetFromDarknet('/home/bot/ros2_ws/yolodeux/yolov3.cfg', '/home/bot/ros2_ws/yolodeux/yolov3.weights')
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




                



                for out in outputs:
                     print(out.shape)
                cropped=[]
                position0=[]
                position1=[]
                def trackbar2(x):
                    confidence = x/100
                    r = r0.copy()
                    for output in np.vstack(outputs):
                        scores = output[5:]
                        classID = np.argmax(scores)
                        if output[4] > confidence:
                            x, y, w, h = output[:4]
                            p0 = [int((x-w/2)*640), int((y-h/2)*640)]
                            p1 = [int((x+w/2)*640), int((y+h/2)*640)]
                            cv2.rectangle(r, p0, p1, 1, 1)
                            position0.append(p0)
                            position1.append(p1)
                            image_cropped=color_image[max(p0[1]-30,0):min(p1[1]+30,640),max(p0[0]-30,0):min(p1[0]+30,640)]
                            cropped.append(image_cropped)  
                    cv2.imshow('papa', cropped[0])   
                    cv2.imshow('blob', r)
                    text = f'Bbox confidence={confidence}'
                    cv2.displayOverlay('blob', text)
                r0 = blob[0, 0, :, :]
                r = r0.copy()
                cv2.imshow('blob', r)
                cv2.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
                trackbar2(50)

#################
                # print(len(cropped))
                if (len(cropped)>0):
                    for i in range(len(cropped)):
                    # Convertir l'image en niveaux de gris
                        gray = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)

                    # Appliquer un seuillage pour détecter les contours
                        ret, thresh = cv2.threshold(gray, 127, 255, 0)

                    # Trouver les contours dans l'image
                        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Dessiner tous les contours sur l'image originale en vert avec une épaisseur de 3 pixels
                        imagedetouree = cv2.drawContours(cropped[i], contours, -1, (0, 255, 0), 1 )
                    #cv2.imshow('ntm',imagedetouree)
                        points=[]

                    # Parcourir la liste des contours et calculer les centres de gravité pour chaque objet
                        for cnt in contours:
                    
                        # Calculer les moments d'images pour l'objet courant
                            moments = cv2.moments(cnt)
                        
                        # Vérifier que le dénominateur du moment m00 est non nul
                            if moments['m00'] != 0:
                            
                            # Calculer les coordonnées x et y du centre de gravité
                                cx = int(moments['m10'] / moments['m00'])
                                cy = int(moments['m01'] / moments['m00'])
                                points.append((cx,cy))
                            
                            # Dessiner un cercle représentant le centre de gravité sur l'image
                                cv2.circle(cropped[i-1], (cx, cy), 5, (0, 0, 255), -1)

                    # Afficher l'image avec les centres de gravité dessinés
                        barycentrex=0
                        barycentrey=0
                        for j in range (len(points)):
                            barycentrex+=points[j][0]
                            barycentrey+=points[j][1]
                        barycentrex= int(barycentrex/(len(points)+1))
                        barycentrey=int(barycentrey/(len(points)+1))
                        cv2.circle(cropped[i], (barycentrex, barycentrey), 7, (255,0 , 0), -1)
                        cv2.imshow('Nathalie', cropped[i])

    ###############

                boxes = []
                boxes2 = []
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
                            x1 = (centerX - (width / 2))
                            y1 = (centerY - (height / 2))

                            
                            x=int(x1)
                            y=int(y1)
                            box2 = [x, y,centerX ,centerY]
                            box = [x, y, int(width), int(height)]
                            boxes.append(box)
                            boxes2.append(box2)
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                            
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indices) > 0:
                    for i in indices.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])     
                        color = [int(c) for c in colors[classIDs[i]]]
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
                        point = (boxes2[i][2], boxes2[i][3])
                        aligned_frames = self.align.process(frames)

                        # Get aligned frames
                        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                        color_frame = aligned_frames.get_color_frame()

                        # Validate that both frames are valid
                        # if not aligned_depth_frame or not color_frame:
                        #     continue

                        depth_map = np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale * 1000
                        dist = aligned_depth_frame.get_distance(int(point[0]),
                                                                int(point[1])) * 1000  # convert to mm
                        # calculate real RGB world coordinates
                        X = dist * (point[0] - self.color_intrinsics.ppx) / self.color_intrinsics.fx
                        Y = dist * (point[1] - self.color_intrinsics.ppy) / self.color_intrinsics.fy
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

                        text = "depth : {:}".format(round(position[2], 5))
                        cv2.putText(color_image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        text2= "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                        cv2.putText(color_image, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
