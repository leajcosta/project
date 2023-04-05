import argparse
import os
import platform
import sys
from pathlib import Path
import pyrealsense2 as rs
import torch
import torch.backends.cudnn as cudnn

#..... Tracker modules......
import skimage
from sort_count import *
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import*
from cv_bridge import CvBridge
#..... Tracker modules......

from sort_count import *
import numpy as np
#...........................


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
class camera(Node):

    def __init__(self):
        
        self.weights=ROOT/'yolov7-seg.pt'  # model.pt path(s)
        self.source=0  # file/dir/URL/glob, 0 for webcam
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=0.7  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=True  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/predict-seg'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.trk = False


    def run(self): 

        #.... Initialize SORT .... 
            
        sort_max_age = 5 
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        sort_tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh) 
        #......................... 

        self.source = str(self.source)
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)


        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size





        # Dataloader
        if webcam:
            if int(self.source) == 0: #realsense
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt)
                bs = len(dataset)  # batch_size
            else: #une webcam
                print("code webcam pas encore fait")
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        count = 0
        for path, im, im0s, dep, vid_cap, s in dataset:
            depth_image = np.asanyarray(dep.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            count +=1
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                self.visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred, out = self.model(im, augment=self.augment, visualize=self.visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                



                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                

                    # Segments
                    if self.save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(6), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w


                    # Mask plotting ----------------------------------------------------------------------------------------

                    if self.trk:
                        #Tracking ----------------------------------------------------
                        dets_to_sort = np.empty((0,6))
                        for x1,y1,x2,y2,conf,detclass in det[:, :6].cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))

                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks =sort_tracker.getTrackers()

                        for track in tracks:
                            annotator.draw_trk(self.line_thickness,track)

                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            annotator.draw_id(bbox_xyxy, identities, categories, names)
                
                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate((det[:, :6])):
                        if self.save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *segj, conf) if self.save_conf else (cls, *segj)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        pt=[]
                        if len(masks)!=0:
                                sx=0
                                sy=0
                                num=0
                                for i in range(int(xyxy[0]), int(xyxy[2]), 3):
                                    for k in range(int(xyxy[1]),int(xyxy[3]), 3):
                                        if masks[j][k][i]==1:
                                            sx+=i
                                            sy+=k
                                            num+=1
                                if num!=0:
                                    pt=[sx/num, sy/num]
                                    dist = dep.get_distance(int(pt[0]), int(pt[1])) * 1000  # convert to mm
                                        # calculate real RGB world coordinates
                                    Z = dist
                                        # calculate  real center of the realsense world coordinates in mm
                                    Z = Z
                                        # calculate  real center of the realsense world coordinates in meter
                                    Z = Z/1000
                                    if save_img or self.save_crop or view_img:  # Add bbox to image
                                        c = int(cls)  # integer class
                                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                                        text = f'{names[c]} {conf:.2f} {round(Z, 5)}'
                                        annotator.box_label(xyxy, text, color=colors(c, True))
                        
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer[i].write(im0)
                #        # cv2.imwrite("depth_{:03}.png".format(count),np.asarray(dep.get_data()))

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT/'yolov7-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--trk', action='store_true', help='Apply Sort Tracking')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main():
    rclpy.init()
    Camera=camera()
    print(Camera.weights)
    check_requirements(exclude=('tensorboard', 'thop'))
    Camera.run()
    rclpy.spin()
    Camera.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
  

