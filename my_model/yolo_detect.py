import os
import sys
import argparse
import glob
import time
import pyttsx3

import cv2
import numpy as np
from ultralytics import YOLO

import threading
import speech_recognition as sr

import pytesseract

import queue

CMD_SCAN   = "SCAN"
CMD_GUIDE  = "GUIDE"
CMD_SELECT = "SELECT"
CMD_READ   = "READ"

COMMAND_VOCAB = {
    CMD_SCAN: [
        "scan", "start scan", "scanning"
    ],
    CMD_GUIDE: [
        "guide", "guidance"
    ],
    CMD_SELECT: [
        "choose", "select", "object", "select object"
    ],
    CMD_READ: [
        "read", "read text", "read it"
    ]
}

COMMAND_COOLDOWN = 2.5  # seconds between commands
last_command_time = 0

#command resolver
def resolve_command(text):
    """
    Converts free-form speech text to a fixed command or None.
    Only matches against defined vocabulary.
    """
    if not text:
        return None

    text = text.lower().strip()

    for command, keywords in COMMAND_VOCAB.items():
        for kw in keywords:
            if kw in text:
                return command

    return None


def command_allowed():
    """
    Enforces cooldown between commands to prevent rapid-fire execution.
    """
    global last_command_time
    now = time.time()
    if now - last_command_time < COMMAND_COOLDOWN:
        return False
    last_command_time = now
    return True

# text to speech engine initialization
engine = pyttsx3.init()
spoken_objects_global = set() 
last_speak_time = 0 
detection_start_time = {}      # to track when each object first appeared
CONFIRMATION_TIME = 1.0   

Frame_Guidance_Cooldown = 1.5   # second between guidance for same obj
last_guidance_time = {}

STATE_SCAN = 0     
STATE_GUIDE = 1        
current_state = STATE_SCAN

COMMAND_TIMEOUT = 5.0

#voice command variables
voice_command = None
voice_command_lock = threading.Lock()

tts_queue = queue.Queue() #queue for text to speech cause multiple overlaping 

is_speaking = False

# tts engine
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

#tts worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

#speak engine that now uses the tts queue
def speak(text):
    tts_queue.put(text)

#voice command listener thread
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    print("Voice command listener started")

    global voice_command, voice_command_lock

    while True:
        with mic as source:
            print("Listening for voice commands...")
            audio = recognizer.listen(source, phrase_time_limit=4)
        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"Voice heard: {text}")
            
            # Resolve to command using closed vocabulary
            cmd = resolve_command(text)
            if cmd:
                print(f"Command resolved: {cmd}")
                with voice_command_lock:
                    voice_command = cmd
            else:
                print("No matching command found - ignored")
                
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Recognition error; {e}")



def do_ocr_on_object(frame, bbox):
    xmin, ymin, xmax, ymax = bbox
    crop_img = frame[ymin:ymax, xmin:xmax]
    # Converting  to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing for better OCR
    gray = cv2.equalizeHist(gray) #increase contrast
    
    gray = cv2.fastNlMeansDenoising(gray, h=10) #denoising
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #thresholding 
    
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    #fallback to original greyscale
    if not text.strip():
        text = pytesseract.image_to_string(gray, config='--psm 6')
    
    return text.strip()

def do_ocr_on_cropped_image(crop_img):
    # Converting to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    # Fallback to original
    if not text.strip():
        text = pytesseract.image_to_string(gray, config='--psm 6')
    
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection")
    parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g., best.pt)')
    parser.add_argument('--source', required=True, help='Image, folder, video file, or webcam index (0)')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--resolution', default=None, help='WxH display resolution, e.g., 640x480')
    parser.add_argument('--record', action='store_true', help='Record video output (requires --resolution)')
    args = parser.parse_args()

    model_path = args.model
    source = args.source
    conf_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    # Check model
    if not os.path.exists(model_path):
        print(f'ERROR: Model file not found: {model_path}')
        sys.exit(1)

    # Load YOLO model
    model = YOLO(model_path)
    labels = model.names

    # Parse resolution
    resize = False
    if user_res:
        resize = True
        resW, resH = map(int, user_res.split('x'))

    # Determine source type
    img_exts = ['.jpg','.jpeg','.png','.bmp']
    vid_exts = ['.mp4','.avi','.mov','.mkv']

    if os.path.isdir(source):
        source_type = 'folder'
        imgs_list = [f for f in glob.glob(f"{source}/*") if os.path.splitext(f)[1].lower() in img_exts]
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in img_exts:
            source_type = 'image'
            imgs_list = [source]
        elif ext in vid_exts:
            source_type = 'video'
        else:
            print(f'File type not supported: {ext}')
            sys.exit(1)
    elif source.isdigit():
        source_type = 'usb'
        cam_idx = int(source)
    else:
        print(f'Invalid source: {source}')
        sys.exit(1)

    # Setup video/camera capture
    if source_type in ['video', 'usb']:
        cap = cv2.VideoCapture(cam_idx if source_type=='usb' else source)
        if resize:
            cap.set(3, resW)
            cap.set(4, resH)
        if record:
            if not resize:
                print("Must specify --resolution to record.")
                sys.exit(1)
            recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
                   (88,159,106), (96,202,231), (159,124,168), (169,162,241),
                   (98,118,150), (172,176,184)]

    fps_buffer = []
    fps_avg_len = 200
    img_count = 0

    global last_speak_time, spoken_objects_global, current_state, voice_command, voice_command_lock

    # Starting the voice command listener thread
    listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listener_thread.start()

    active_object = None
    active_object_bbox = None

    while True:
        t_start = time.perf_counter()

        # Load frame
        if source_type in ['image','folder']:
            if img_count >= len(imgs_list):
                print("All images processed.")
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1
        elif source_type in ['video','usb']:
            ret, frame = cap.read()
            if not ret:
                print("Video/camera ended or failed.")
                break

        # Resize frame
        if resize:
            frame = cv2.resize(frame, (resW,resH))

        frame_width = frame.shape[1]
        left_zone = frame_width / 3
        right_zone = 2 * frame_width / 3

        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        obj_count = 0

        # Draw detections
        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            class_idx = int(det.cls.item())
            conf = det.conf.item()
            if conf < conf_thresh:
                continue
            color = bbox_colors[class_idx % 10]
            classname = labels[class_idx]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {conf:.2f}'
            cv2.putText(frame, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            obj_count += 1

        # Announce detected object
        current_frame_objects = set()
        for det in detections:
            classname = labels[int(det.cls.item())]
            current_frame_objects.add(classname)

        current_time = time.time()

        # Adding new objects with the current timestamp
        for obj in current_frame_objects:
            if obj not in detection_start_time:
                detection_start_time[obj] = current_time

        # Removing objects that leave the frame
        for obj in list(detection_start_time.keys()):
            if obj not in current_frame_objects:
                detection_start_time.pop(obj)

        # Speak for new objects only after confirming.
        if current_state == STATE_SCAN:
            for obj, start_time in detection_start_time.items():
                if obj not in spoken_objects_global and (current_time - start_time) >= CONFIRMATION_TIME:
                    # Find its bounding box in current detections
                    for det in detections:
                        classname = labels[int(det.cls.item())]
                        if classname == obj:
                            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                            xmin, ymin, xmax, ymax = xyxy
                            x_center = (xmin + xmax) / 2
                            if x_center < left_zone:
                                position = "left"
                            elif x_center > right_zone:
                                position = "right"
                            else:
                                position = "center"
                            break

                    # Speak object + position
                    speak(f"Detected {obj} on the {position}")

                    spoken_objects_global.add(obj)

        # Removeing the objects that leave the frame
        spoken_objects_global = spoken_objects_global.intersection(current_frame_objects)

        if current_state == STATE_GUIDE:
            now = time.time()
            for det in detections:
                classname = labels[int(det.cls.item())]
                conf = det.conf.item()
                if conf < conf_thresh:
                    continue
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy

                frame_height, frame_width = frame.shape[:2]
                bbox_area = (xmax - xmin) * (ymax - ymin)
                frame_area = frame_width * frame_height
                area_ratio = bbox_area / frame_area

                last_time = last_guidance_time.get(classname, 0)

                if now - last_time > Frame_Guidance_Cooldown:
                    if area_ratio < 0.20:
                        speak(f"Move the {classname} closer.")
                    elif area_ratio > 0.55:
                        speak(f"Move the {classname} slightly away.")
                    else:
                        speak(f"Hold steady on the {classname}.")
                    last_guidance_time[classname] = now

        # Draw FPS and object count
        if source_type in ['video','usb']:
            avg_fps = np.mean(fps_buffer) if fps_buffer else 0
            cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f'Objects: {obj_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        state_text = "STATE: SCAN" if current_state == STATE_SCAN else "STATE: GUIDE"
        state_color = (0, 255, 0) if current_state == STATE_SCAN else (0, 0, 255)
        cv2.putText(frame, state_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        if current_state == STATE_GUIDE and active_object_bbox is not None:
            xmin, ymin, xmax, ymax = active_object_bbox
            text = do_ocr_on_object(frame, active_object_bbox)
            if text:
                cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("YOLO Detection", frame)

        if record:
            recorder.write(frame)

        key = cv2.waitKey(1 if source_type in ['video','usb'] else 0) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('p'), ord('P')]:
            cv2.imwrite('capture.png', frame)
        elif key == ord('g'):  # press 'g' to enter guide mode
            current_state = STATE_GUIDE
            last_guidance_time.clear()
            speak("Entering guide mode.")
        elif key == ord('s'):  # press 's' to return to scan mode
            current_state = STATE_SCAN
            last_guidance_time.clear()
            speak("Returning to scan mode.")
        elif key == ord('r'):
            if current_state == STATE_GUIDE and active_object and active_object_bbox is not None:
                xmin, ymin, xmax, ymax = active_object_bbox
                bbox_area = (xmax - xmin) * (ymax - ymin)
                frame_area = frame.shape[0] * frame.shape[1]
                area_ratio = bbox_area / frame_area

                if area_ratio > 0.5:
                    text = do_ocr_on_object(frame, active_object_bbox)
                    if text:
                        speak(f"Reading text: {text}")
                    else:
                        speak("No text detected.")
                else:
                    speak("Please bring the object closer to read the text.")
            else:
                speak("Please enter guide mode and select an object to read.")

        # deterministic command handeling
        with voice_command_lock:
            cmd = voice_command
            voice_command = None  # reset after reading

        if cmd and command_allowed():

            if cmd == CMD_GUIDE:
                current_state = STATE_GUIDE
                last_guidance_time.clear()
                speak("Guide mode")

            elif cmd == CMD_SCAN:
                current_state = STATE_SCAN
                last_guidance_time.clear()
                speak("Scan mode")

            elif cmd == CMD_SELECT:
                if detections and len(detections) > 0:
                    # Select largest object by area
                    active_object_det = max(
                        detections,
                        key=lambda d: (
                            (d.xyxy[0][2] - d.xyxy[0][0]) *
                            (d.xyxy[0][3] - d.xyxy[0][1])
                        )
                    )
                    active_object_bbox = active_object_det.xyxy.cpu().numpy().squeeze().astype(int)
                    active_object = labels[int(active_object_det.cls.item())]

                    current_state = STATE_GUIDE
                    speak(f"{active_object} selected")
                else:
                    speak("No objects detected")

            elif cmd == CMD_READ:
                print(f" READ command triggered")
                print(f" current_state = {current_state} (STATE_GUIDE={STATE_GUIDE})")
                print(f" active_object = {active_object}")
                print(f" active_object_bbox = {active_object_bbox}")
                
                if current_state == STATE_GUIDE and active_object_bbox is not None:
                    # Check if object is large enough (close enough to camera)
                    current_bbox = None
                    for det in detections:
                        if labels[int(det.cls.item())] == active_object:
                            current_bbox = det.xyxy.cpu().numpy().squeeze().astype(int)
                            break

                    if current_bbox is None:
                        speak("Object not visible")
                        print(" Object not visible in current frame")
                        continue 

                    xmin, ymin, xmax, ymax = current_bbox
                    bbox_area = (xmax - xmin) * (ymax - ymin)
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = bbox_area / frame_area
                    
                    print(f" Area ratio: {area_ratio:.3f}")
                    
                    if area_ratio < 0.20:
                        speak("Please bring the object closer to read")
                        print(f" Object too far (area_ratio={area_ratio:.3f})")

                    elif area_ratio > 0.55:
                        speak("Move the object slightly away")
                        print(f" Object too close (area_ratio={area_ratio:.3f})")

                    else:
                        speak("Reading... Hold steady")
                        print(" Capturing multiple frames for better OCR...")
                        
                        # Captureing multiple frames for better OCR
                        ocr_results = []
                        for attempt in range(5):
                            # Getting fresh frame
                            if source_type in ['video','usb']:
                                ret, fresh_frame = cap.read()
                                if not ret:
                                    break
                                if resize:
                                    fresh_frame = cv2.resize(fresh_frame, (resW, resH))
                                
                                # Running YOLO to the update bbox
                                fresh_results = model(fresh_frame, verbose=False)
                                fresh_detections = fresh_results[0].boxes
                                
                                # Finding the same object in new frame
                                for det in fresh_detections:
                                    if labels[int(det.cls.item())] == active_object:
                                        fresh_bbox = det.xyxy.cpu().numpy().squeeze().astype(int)
                                        text = do_ocr_on_object(fresh_frame, fresh_bbox)
                                        if text and len(text) > 3:  # Only keep meaningful results
                                            ocr_results.append(text)
                                            print(f"[] Attempt {attempt+1}: '{text}' (len={len(text)})")
                                        break
                            else:
                                # For static images, just trying once
                                text = do_ocr_on_object(frame, active_object_bbox)
                                if text:
                                    ocr_results.append(text)
                                break
                        
                        # Choosing the longest result
                        if ocr_results:
                            best_text = max(ocr_results, key=len)
                            print(f"Best OCR result: '{best_text}'")
                            
                            speak(f"Reading text: {best_text}")
                            print(f"Speaking: Reading text: {best_text}")
                        else:
                            speak("No text detected. Try holding the object steady")
                            print(f"Speaking: No text detected")
                else:
                    speak("Select an object first")
                    print(f"Speaking: Select an object first")

        # Update FPS buffer
        t_stop = time.perf_counter()
        fps_buffer.append(1/(t_stop-t_start))
        if len(fps_buffer) > fps_avg_len:
           fps_buffer.pop(0)

    # Cleanup
    if source_type in ['video','usb']:
        cap.release()
    if record:
        recorder.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {np.mean(fps_buffer):.2f}")

if __name__ == "__main__":
    main()