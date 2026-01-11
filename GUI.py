import os
import sys
import argparse
import glob
import time
import pyttsx3
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import speech_recognition as sr
import pytesseract
import queue

CMD_SCAN = "SCAN"
CMD_GUIDE = "GUIDE"
CMD_SELECT = "SELECT"
CMD_READ = "READ"

COMMAND_VOCAB = {
    CMD_SCAN: ["scan", "start scan", "scanning"],
    CMD_GUIDE: ["guide", "guidance"],
    CMD_SELECT: ["choose", "select", "object", "select object"],
    CMD_READ: ["read", "read text", "read it"]
}

COMMAND_COOLDOWN = 2.5
STATE_SCAN = 0
STATE_GUIDE = 1
CONFIRMATION_TIME = 1.0
FRAME_GUIDANCE_COOLDOWN = 1.5

#global variables
last_command_time = 0
spoken_objects_global = set()
detection_start_time = {}
last_guidance_time = {}
voice_command = None
voice_command_lock = threading.Lock()
tts_queue = queue.Queue()

# converting speech to command
def resolve_command(text):
    if not text:
        return None
    text = text.lower().strip()
    for command, keywords in COMMAND_VOCAB.items():
        for kw in keywords:
            if kw in text:
                return command
    return None

#cooldown between commands
def command_allowed():
    global last_command_time
    now = time.time()
    if now - last_command_time < COMMAND_COOLDOWN:
        return False
    last_command_time = now
    return True

# tts worker
def tts_worker():
    engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()
#tts worker thresd
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)

#command listerner
def listen_for_commands(app):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    global voice_command
    
    while app.running:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=4)
            text = recognizer.recognize_google(audio).lower()
            
            cmd = resolve_command(text)
            if cmd:
                with voice_command_lock:
                    voice_command = cmd
                app.log_command(f"Voice: {text} → {cmd}")
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            app.log_command(f"Recognition error: {e}")
        except Exception as e:
            if app.running:
                app.log_command(f"Listener error: {e}")

# ocr
def do_ocr_on_bbox(frame, bbox):
    xmin, ymin, xmax, ymax = bbox
    crop_img = frame[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    if not text.strip():
        text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip()

# gui 
class VisionAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Assistant System")
        self.root.geometry("1200x800")
        
        # Application state
        self.running = True
        self.capturing = False
        self.model = None
        self.labels = None
        self.cap = None
        self.current_state = STATE_SCAN
        self.active_object = None
        self.active_object_bbox = None
        self.conf_threshold = 0.5
        self.source_type = None
        self.resize = False
        self.resW, self.resH = 640, 480
        
        # FPS tracking
        self.fps_buffer = []
        self.fps_avg_len = 200
        
        # Detection tracking
        global spoken_objects_global, detection_start_time, last_guidance_time
        spoken_objects_global = set()
        detection_start_time = {}
        last_guidance_time = {}
        
        # Colors
        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
                           (88,159,106), (96,202,231), (159,124,168), (169,162,241),
                           (98,118,150), (172,176,184)]
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        left_panel = tk.Frame(main_frame, width=300, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg="#000000")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        #left panel stuff
        # Title
        title_label = tk.Label(left_panel, text="Vision Assistant", font=("Arial", 16, "bold"), bg="#f0f0f0") 
        title_label.pack(pady=10)
        
        # Model Selection menu
        model_frame = tk.LabelFrame(left_panel, text="Model", bg="#f0f0f0")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill=tk.X, padx=5, pady=5)
        self.model_label = tk.Label(model_frame, text="No model loaded", bg="#f0f0f0", fg="red")
        self.model_label.pack(padx=5, pady=2)
        
        # Source Selection menu
        source_frame = tk.LabelFrame(left_panel, text="Video Source", bg="#f0f0f0")
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(source_frame, text="Webcam", command=self.start_webcam).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(source_frame, text="Video File", command=self.load_video).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(source_frame, text="Stop", command=self.stop_capture, bg="#ff6b6b").pack(fill=tk.X, padx=5, pady=2)
        
        # Mode Selection
        mode_frame = tk.LabelFrame(left_panel, text="Operating Mode", bg="#f0f0f0")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.scan_btn = tk.Button(mode_frame, text="SCAN Mode", command=self.set_scan_mode, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.scan_btn.pack(fill=tk.X, padx=5, pady=2)
        self.guide_btn = tk.Button(mode_frame, text="GUIDE Mode", command=self.set_guide_mode, bg="#808080", fg="white", font=("Arial", 10, "bold"))
        self.guide_btn.pack(fill=tk.X, padx=5, pady=2)
        
        # Settings
        settings_frame = tk.LabelFrame(left_panel, text="Settings", bg="#f0f0f0")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(settings_frame, text="Confidence Threshold:", bg="#f0f0f0").pack(anchor=tk.W, padx=5)
        self.conf_scale = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, command=self.update_confidence)
        self.conf_scale.set(0.5)
        self.conf_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Actions
        actions_frame = tk.LabelFrame(left_panel, text="Actions", bg="#f0f0f0")
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(actions_frame, text="Select Object", command=self.select_object).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(actions_frame, text="Read Text", command=self.read_text).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(actions_frame, text="Capture Image", command=self.capture_image).pack(fill=tk.X, padx=5, pady=2)
        
        # Detected Objects List
        objects_frame = tk.LabelFrame(left_panel, text="Detected Objects", bg="#f0f0f0")
        objects_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.objects_listbox = tk.Listbox(objects_frame, height=6)
        self.objects_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.objects_listbox.bind('<<ListboxSelect>>', self.on_object_select)
        
        # Voice Command Log
        log_frame = tk.LabelFrame(left_panel, text="Command Log", bg="#f0f0f0")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
        # Rigth pannel stuff
        # Video canvas
        self.canvas = tk.Canvas(right_panel, bg="#000000")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # OCR Results
        ocr_frame = tk.LabelFrame(right_panel, text="OCR Results", bg="#f0f0f0")
        ocr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ocr_text = scrolledtext.ScrolledText(ocr_frame, height=4, wrap=tk.WORD)
        self.ocr_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # status bar
        status_bar = tk.Frame(self.root, bg="#333333", height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(status_bar, text="Status: Ready", bg="#333333", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(status_bar, text="FPS: 0", bg="#333333", fg="white")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Keyboard bindings
        self.root.bind('<g>', lambda e: self.set_guide_mode())
        self.root.bind('<s>', lambda e: self.set_scan_mode())
        self.root.bind('<r>', lambda e: self.read_text())
        self.root.bind('<p>', lambda e: self.capture_image())
        self.root.bind('<q>', lambda e: self.on_closing())
        
        # Starting voice listener
        self.voice_thread = threading.Thread(target=listen_for_commands, args=(self,), daemon=True)
        self.voice_thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLO Model",filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.labels = self.model.names
                self.model_label.config(text=os.path.basename(model_path), fg="green")
                self.log_command(f"Model loaded: {os.path.basename(model_path)}")
                speak("Model loaded successfully")
            except Exception as e:
                self.log_command(f"Error loading model: {e}")
                speak("Error loading model")
    
    def start_webcam(self):
        if not self.model:
            self.log_command("Please load a model first")
            speak("Please load a model first")
            return
        
        self.source_type = 'webcam'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.resW)
        self.cap.set(4, self.resH)
        self.capturing = True
        self.log_command("Webcam started")
        speak("Webcam started")
        self.process_video()
    
    def load_video(self):
        if not self.model:
            self.log_command("Please load a model first")
            speak("Please load a model first")
            return
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if video_path:
            self.source_type = 'video'
            self.cap = cv2.VideoCapture(video_path)
            self.capturing = True
            self.log_command(f"Video loaded: {os.path.basename(video_path)}")
            speak("Video loaded")
            self.process_video()
    
    def stop_capture(self):
        self.capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log_command("Capture stopped")
        speak("Capture stopped")
    
    def set_scan_mode(self):
        self.current_state = STATE_SCAN
        self.scan_btn.config(bg="#4CAF50")
        self.guide_btn.config(bg="#808080")
        last_guidance_time.clear()
        self.log_command("Mode: SCAN")
        speak("Scan mode")
    
    def set_guide_mode(self):
        self.current_state = STATE_GUIDE
        self.guide_btn.config(bg="#4CAF50")
        self.scan_btn.config(bg="#808080")
        last_guidance_time.clear()
        self.log_command("Mode: GUIDE")
        speak("Guide mode")
    
    def update_confidence(self, value):
        self.conf_threshold = float(value)
    
    def select_object(self):
        # This can simply be called by button or voice command
        self.handle_select_command()
    
    def read_text(self):
        # This can simply be called by button or voice command
        self.handle_read_command()
    
    def capture_image(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            cv2.imwrite('capture.png', self.current_frame)
            self.log_command("Image saved as capture.png")
            speak("Image captured")
    
    def on_object_select(self, event):
        selection = self.objects_listbox.curselection()
        if selection and hasattr(self, 'current_detections'):
            idx = selection[0]
            if idx < len(self.current_detections):
                det = self.current_detections[idx]
                self.active_object_bbox = det['bbox']
                self.active_object = det['class']
                self.current_state = STATE_GUIDE
                self.set_guide_mode()
                speak(f"{self.active_object} selected")
    
    def log_command(self, text):
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)
        # Keeping only last 50 lines
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 50:
            self.log_text.delete('1.0', '2.0')
    
    def handle_select_command(self):
        if hasattr(self, 'current_detections') and self.current_detections:
            # Selecting largest object
            largest = max(self.current_detections, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
            self.active_object_bbox = largest['bbox']
            self.active_object = largest['class']
            self.current_state = STATE_GUIDE
            self.set_guide_mode()
            speak(f"{self.active_object} selected")
        else:
            speak("No objects detected")
    
    def handle_read_command(self):
        if self.current_state == STATE_GUIDE and self.active_object_bbox is not None:
            if not hasattr(self, 'current_frame') or self.current_frame is None:
                speak("No frame available")
                return
            
            # Checking if object still visible
            current_bbox = None
            if hasattr(self, 'current_detections'):
                for det in self.current_detections:
                    if det['class'] == self.active_object:
                        current_bbox = det['bbox']
                        break
            
            if current_bbox is None:
                speak("Object not visible")
                return
            
            xmin, ymin, xmax, ymax = current_bbox
            bbox_area = (xmax - xmin) * (ymax - ymin)
            frame_area = self.current_frame.shape[0] * self.current_frame.shape[1]
            area_ratio = bbox_area / frame_area
            
            if area_ratio < 0.20:
                speak("Please bring the object closer to read")
            elif area_ratio > 0.55:
                speak("Move the object slightly away")
            else:
                speak("Reading... Hold steady")
                
                # Putting the same Multi-frame OCR for better results. as prev done in yolo_detect.
                ocr_results = []
                for attempt in range(5):
                    if self.cap and self.capturing:
                        ret, fresh_frame = self.cap.read()
                        if not ret:
                            break
                        
                        fresh_results = self.model(fresh_frame, verbose=False)
                        fresh_detections = fresh_results[0].boxes
                        
                        for det in fresh_detections:
                            if self.labels[int(det.cls.item())] == self.active_object:
                                fresh_bbox = det.xyxy.cpu().numpy().squeeze().astype(int)
                                text = do_ocr_on_bbox(fresh_frame, fresh_bbox)
                                if text and len(text) > 3:
                                    ocr_results.append(text)
                                break
                    else:
                        text = do_ocr_on_bbox(self.current_frame, self.active_object_bbox)
                        if text:
                            ocr_results.append(text)
                        break
                
                if ocr_results:
                    best_text = max(ocr_results, key=len)
                    self.ocr_text.delete('1.0', tk.END)
                    self.ocr_text.insert('1.0', best_text)
                    speak(f"Reading text: {best_text}")
                else:
                    speak("No text detected. Try holding the object steady")
        else:
            speak("Select an object first")
    
    def process_video(self):
        if not self.capturing or self.cap is None:
            return
        
        t_start = time.perf_counter()
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_capture()
            return
        
        self.current_frame = frame.copy()
        frame_display = frame.copy()
        
        frame_height, frame_width = frame.shape[:2]
        left_zone = frame_width / 3
        right_zone = 2 * frame_width / 3
        
        # YOLO inference
        results = self.model(frame, verbose=False)
        detections = results[0].boxes
        
        # Storing detections for selection
        self.current_detections = []
        current_frame_objects = set()
        
        # Drawing detections
        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            class_idx = int(det.cls.item())
            conf = det.conf.item()
            
            if conf < self.conf_threshold:
                continue
            
            classname = self.labels[class_idx]
            current_frame_objects.add(classname)
            
            self.current_detections.append({
                'class': classname,
                'conf': conf,
                'bbox': [xmin, ymin, xmax, ymax]
            })
            
            color = self.bbox_colors[class_idx % 10]
            cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {conf:.2f}'
            cv2.putText(frame_display, label, (xmin, ymin-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        # Object announcement logic during SCAN mode
        current_time = time.time()
        
        for obj in current_frame_objects:
            if obj not in detection_start_time:
                detection_start_time[obj] = current_time
        
        for obj in list(detection_start_time.keys()):
            if obj not in current_frame_objects:
                detection_start_time.pop(obj)
        
        if self.current_state == STATE_SCAN:
            for obj, start_time in detection_start_time.items():
                if obj not in spoken_objects_global and (current_time - start_time) >= CONFIRMATION_TIME:
                    # Finding position
                    for det in self.current_detections:
                        if det['class'] == obj:
                            xmin, ymin, xmax, ymax = det['bbox']
                            x_center = (xmin + xmax) / 2
                            if x_center < left_zone:
                                position = "left"
                            elif x_center > right_zone:
                                position = "right"
                            else:
                                position = "center"
                            break
                    
                    speak(f"Detected {obj} on the {position}")
                    spoken_objects_global.add(obj)
        
        spoken_objects_global = spoken_objects_global.intersection(current_frame_objects)
        
        # Guidance mode
        if self.current_state == STATE_GUIDE:
            for det in self.current_detections:
                classname = det['class']
                xmin, ymin, xmax, ymax = det['bbox']
                
                bbox_area = (xmax - xmin) * (ymax - ymin)
                frame_area = frame_width * frame_height
                area_ratio = bbox_area / frame_area
                
                last_time = last_guidance_time.get(classname, 0)
                
                if current_time - last_time > FRAME_GUIDANCE_COOLDOWN:
                    if area_ratio < 0.20:
                        speak(f"Move the {classname} closer.")
                    elif area_ratio > 0.55:
                        speak(f"Move the {classname} slightly away.")
                    else:
                        speak(f"Hold steady on the {classname}.")
                    last_guidance_time[classname] = current_time
        
        # Updating objects listbox
        self.objects_listbox.delete(0, tk.END)
        for det in self.current_detections:
            self.objects_listbox.insert(tk.END, f"{det['class']}: {det['conf']:.2f}")
        
        # Drawing state and info
        state_text = "SCAN" if self.current_state == STATE_SCAN else "GUIDE"
        state_color = (0, 255, 0) if self.current_state == STATE_SCAN else (0, 0, 255)
        cv2.putText(frame_display, f"Mode: {state_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame_display, f"Objects: {len(self.current_detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Handling the voice commands
        global voice_command
        with voice_command_lock:
            cmd = voice_command
            voice_command = None
        
        if cmd and command_allowed():
            if cmd == CMD_GUIDE:
                self.set_guide_mode()
            elif cmd == CMD_SCAN:
                self.set_scan_mode()
            elif cmd == CMD_SELECT:
                self.handle_select_command()
            elif cmd == CMD_READ:
                self.handle_read_command()
        
        # Displaying the frame
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resizing to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=imgtk)
        self.canvas.imgtk = imgtk
        
        # Updating FPS
        t_stop = time.perf_counter()
        self.fps_buffer.append(1/(t_stop - t_start))
        if len(self.fps_buffer) > self.fps_avg_len:
            self.fps_buffer.pop(0)
        
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
        self.status_label.config(text=f"Status: Capturing | Objects: {len(self.current_detections)}")
        
        # Scheduling the next frame
        self.root.after(10, self.process_video)
    
    def on_closing(self):
        self.running = False
        self.capturing = False
        if self.cap:
            self.cap.release()
        tts_queue.put(None)  # Stoping TTS worker
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VisionAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()