import cv2
import numpy as np
from ultralytics import YOLO
import os
import easyocr
import re
from datetime import datetime
import time
import platform
from face_recog_mod import FaceRecognizer
from database import Database

# Deteksi sistem operasi
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_RASPBERRY_PI = IS_LINUX and os.path.exists('/proc/device-tree/model')

if IS_RASPBERRY_PI:
    print("✅ Running on Raspberry Pi")
elif IS_LINUX:
    print("✅ Running on Linux")
elif IS_WINDOWS:
    print("✅ Running on Windows")

# Import Raspberry Pi Controller
try:
    from rasp import RaspberryPiController
    RASP_AVAILABLE = True
except ImportError:
    print("⚠️ Raspberry Pi Controller not available (rasp.py not found or RPi.GPIO not installed)")
    RASP_AVAILABLE = False

# --- KONFIGURASI ---
MODEL_PATH = "best.pt"
IMAGE_PATH = "plat3.jpg" 
OUTPUT_PATH = "output_plat3.jpg"
PLATES_FOLDER = "detected_plates"
CONFIDENCE_THRESHOLD = 0.5
CAPTURE_DELAY = 2  # Delay 2 detik setelah deteksi plat sebelum capture

# --- CAMERA CONFIG ---
# Pilih salah satu:
# - 0, 1, 2, ... (index) untuk auto-detect
# - "/dev/video0", "/dev/video1", ... (path) untuk device spesifik di Linux
# Gunakan device path jika ada multiple camera atau auto-detect bermasalah
CAMERA_DEVICE = "/dev/video0" if IS_LINUX else 0  # Auto-select based on OS

# --- PERFORMANCE MODE ---
# "HIGH" = Kualitas terbaik (PC/Laptop dengan GPU)
# "BALANCED" = Seimbang (Raspberry Pi 4/5)
# "LOW" = Performa maksimal (Raspberry Pi 3 atau USB webcam)
PERFORMANCE_MODE = "LOW" if IS_RASPBERRY_PI else "BALANCED"

# Performance settings based on mode
if PERFORMANCE_MODE == "HIGH":
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    OCR_SCALE = 2.0
    PROCESS_EVERY_N_FRAMES = 1
    STABLE_FRAMES_REQUIRED = 3
elif PERFORMANCE_MODE == "BALANCED":
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 20
    OCR_SCALE = 1.8
    PROCESS_EVERY_N_FRAMES = 2
    STABLE_FRAMES_REQUIRED = 3
else:  # LOW
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 10
    OCR_SCALE = 1.5
    PROCESS_EVERY_N_FRAMES = 3
    STABLE_FRAMES_REQUIRED = 2

# --- DEBUG & TESTING ---
DEBUG_MODE = False  # Set False untuk production (mengurangi output log)
CAMERA_TEST_MODE = False  # Test kamera saat startup
HEADLESS_MODE = os.environ.get('DISPLAY') is None if IS_LINUX else False  # Auto-detect headless

# Inisialisasi
print("Initializing EasyOCR...")
# Auto-detect GPU availability (CUDA for NVIDIA, typically not available on Raspberry Pi)
try:
    if IS_RASPBERRY_PI:
        print("  Using CPU mode (Raspberry Pi detected)")
        reader = easyocr.Reader(['en'], gpu=False)
    else:
        # Try GPU first, fallback to CPU if not available
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            print("  Using GPU acceleration")
        except:
            print("  GPU not available, using CPU mode")
            reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"  Warning: EasyOCR initialization issue: {e}")
    print("  Falling back to CPU mode")
    reader = easyocr.Reader(['en'], gpu=False)

print("Initializing Face Recognizer...")
face_recognizer = FaceRecognizer()

print("Initializing Database...")
db = Database()

# Initialize Raspberry Pi Controller if available
if RASP_AVAILABLE:
    print("Initializing Raspberry Pi Controller...")
    rasp_controller = RaspberryPiController()
    if not rasp_controller.is_initialized:
        print("⚠️ Raspberry Pi Controller initialization failed!")
        rasp_controller = None
else:
    rasp_controller = None
    print("⚠️ Running without Raspberry Pi hardware control")

# ===== DEBUGGING FUNCTIONS =====
def debug_log(message, level="INFO"):
    """Print debug messages only if DEBUG_MODE is enabled"""
    if not DEBUG_MODE:
        return
    
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "DEBUG": "🔍",
        "CAMERA": "📷",
        "PROCESS": "⚙️"
    }
    icon = icons.get(level, "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {message}")

def test_camera(camera_device):
    """Test kamera dan return status"""
    print("\n" + "=" * 50)
    print("  CAMERA CONNECTIVITY TEST")
    print("=" * 50)
    
    try:
        print(f"📷 Testing camera {camera_device}...")
        
        # Try opening camera with appropriate backend
        if IS_WINDOWS:
            debug_log("Using DirectShow backend (Windows)", "DEBUG")
            cap = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW) if isinstance(camera_device, int) else cv2.VideoCapture(camera_device)
        elif IS_LINUX:
            debug_log("Using V4L2 backend (Linux)", "DEBUG")
            cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2) if isinstance(camera_device, int) else cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        else:
            debug_log("Using default backend", "DEBUG")
            cap = cv2.VideoCapture(camera_device)
        
        if not cap.isOpened():
            print("❌ FAILED: Camera cannot be opened")
            if IS_LINUX:
                print("   💡 Tip: Check if user is in 'video' group")
                print("      sudo usermod -a -G video $USER")
                print("   💡 Check available cameras: ls -l /dev/video*")
            return False
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        debug_log(f"Camera resolution: {int(width)}x{int(height)}", "DEBUG")
        debug_log(f"Camera FPS: {int(fps)}", "DEBUG")
        
        # Try to read a frame
        print("📸 Attempting to capture test frame...")
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("❌ FAILED: Cannot read frame from camera")
            cap.release()
            return False
        
        print(f"✅ SUCCESS: Camera working properly")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Color channels: {frame.shape[2] if len(frame.shape) > 2 else 1}")
        
        # Save test frame if in debug mode
        if DEBUG_MODE:
            test_image_path = "camera_test.jpg"
            cv2.imwrite(test_image_path, frame)
            debug_log(f"Test frame saved to {test_image_path}", "SUCCESS")
        
        cap.release()
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        print("=" * 50)
        return False

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = YOLO(model_path)
    return model

def preprocess_plate(plate_img):
    """
    Preprocessing dengan optimasi berdasarkan performance mode
    """
    # 1. Upscaling (gunakan scale dari performance config)
    h, w = plate_img.shape[:2]
    scale = OCR_SCALE
    plate_img = cv2.resize(plate_img, (w*int(scale), h*int(scale)), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale & Contrast
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Invert jika gelap (agar tulisan hitam latar putih, atau sebaliknya yang kontras)
    if np.mean(gray) < 100:
        gray = cv2.bitwise_not(gray)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Blur sedikit untuk noise
    processed = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return processed


def read_plate_text(plate_img):
    """
    LOGIKA BARU: Filter berdasarkan Posisi Y (Height)
    OCR membaca dari KIRI ke KANAN berdasarkan koordinat X
    """
    try:
        debug_log("Starting preprocessing...", "PROCESS")
        processed_img = preprocess_plate(plate_img)
        img_h, img_w = processed_img.shape[:2]
        debug_log(f"Preprocessed image size: {img_w}x{img_h}", "DEBUG")
        
        # Save preprocessed image for debugging
        if DEBUG_MODE:
            cv2.imwrite("debug_preprocessed.jpg", processed_img)
            debug_log("Preprocessed image saved to debug_preprocessed.jpg", "DEBUG")

        debug_log("Running EasyOCR... (this may take a while)", "PROCESS")
        # detail=1 memberikan output: [ [[x1,y1],[x2,y2]..], "teks", confidence ]
        results = reader.readtext(
            processed_img,
            detail=1, 
            paragraph=False, # False agar kita dapat koordinat per kata
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            mag_ratio=1.5
        )
        
        debug_log(f"EasyOCR found {len(results)} text regions", "DEBUG")
        
        if not results:
            debug_log("No text detected by OCR", "WARNING")
            return ""

        valid_texts = []
        all_texts = []  # Simpan semua teks sebagai fallback
        
        for idx, (bbox, text, prob) in enumerate(results):
            debug_log(f"  Text {idx+1}: '{text}' (confidence: {prob:.2f})", "DEBUG")
            
            # bbox = [[tl], [tr], [br], [bl]]
            # Ambil koordinat Y bawah (bottom-left dan bottom-right)
            y_bottom = (bbox[2][1] + bbox[3][1]) / 2
            
            # Ambil koordinat X kiri (untuk sorting kiri ke kanan)
            x_left = (bbox[0][0] + bbox[3][0]) / 2
            
            # Simpan semua teks untuk fallback
            all_texts.append((x_left, text))
            
            # LOGIKA FILTER:
            # Jika posisi Y teks berada di 85% ke bawah dari tinggi gambar,
            # kemungkinan besar itu adalah masa berlaku (bulan/tahun).
            # Kita hanya ambil teks yang ada di bagian ATAS (0 - 85% tinggi).
            limit_line = img_h * 0.85
            
            if y_bottom < limit_line:
                # Simpan dengan koordinat X untuk sorting
                valid_texts.append((x_left, text))
                debug_log(f"    ✓ Accepted (Y: {y_bottom:.0f} < {limit_line:.0f})", "DEBUG")
            else:
                print(f"   -> Mengabaikan teks di area bawah: '{text}'")

        # Jika tidak ada valid_texts, gunakan semua teks sebagai fallback
        if not valid_texts and all_texts:
            debug_log("No texts passed filter, using all texts as fallback", "WARNING")
            valid_texts = all_texts
        
        if not valid_texts:
            debug_log("No valid texts found after filtering", "WARNING")
            return ""

        # Sort berdasarkan koordinat X (KIRI ke KANAN)
        valid_texts.sort(key=lambda x: x[0])
        debug_log(f"Sorted texts (left to right): {[t[1] for t in valid_texts]}", "DEBUG")
        
        # Gabungkan teks yang sudah diurutkan dari kiri ke kanan
        # Ensure text is string (handle if it's a list)
        full_text = "".join([str(t[1]) if not isinstance(t[1], str) else t[1] for t in valid_texts]).upper()
        debug_log(f"Combined text: '{full_text}'", "DEBUG")
        
        # --- FINAL CLEANING DENGAN REGEX ---
        # Pola: 1-2 Huruf, 1-4 Angka, 1-3 Huruf
        # Contoh: B 1308 RFO (abaikan sisanya)
        match = re.search(r"([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})", full_text)
        
        if match:
            # Format ulang dengan spasi
            final_text = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            debug_log(f"Regex matched: '{final_text}'", "SUCCESS")
            return final_text
        
        # Jika regex gagal tapi ada teks, kembalikan apa adanya (dibersihkan)
        cleaned = "".join(c for c in full_text if c.isalnum())
        debug_log(f"Regex failed, returning cleaned text: '{cleaned}'", "WARNING")
        return cleaned

    except Exception as e:
        debug_log(f"OCR Exception: {e}", "ERROR")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return ""

def save_plate_image(plate_img, plate_text):
    if not os.path.exists(PLATES_FOLDER):
        os.makedirs(PLATES_FOLDER)
    timestamp = datetime.now().strftime("%H%M%S")
    clean_text = plate_text.replace(" ", "")
    filename = f"{clean_text}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(PLATES_FOLDER, filename), plate_img)

def detect_plate_from_frame(model, frame, confidence=0.5):
    """
    Mendeteksi plat dari frame dan mengembalikan hasil deteksi
    Returns: (detected, plate_text, plate_crop, bbox) 
    """
    height, width = frame.shape[:2]
    results = model(frame, conf=confidence, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop dengan padding sedikit
            pad = 5
            crop = frame[max(0,y1-pad):min(height,y2+pad), max(0,x1-pad):min(width,x2+pad)]
            
            if crop.size > 0:
                # Return koordinat dan crop, belum proses OCR
                return True, None, crop, (x1, y1, x2, y2)
    
    return False, None, None, None

def process_plate_ocr(plate_crop):
    """
    Memproses gambar plat untuk membaca teksnya
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    
    text = read_plate_text(plate_crop)
    return text

def detect_license_plates_realtime(model, confidence=0.5):
    """
    Deteksi plat nomor secara realtime menggunakan webcam.
    Setelah mendeteksi plat, delay 3 detik lalu capture gambar untuk OCR.
    Setelah itu webcam mati dan memulai face recognition.
    """
    print("\n" + "=" * 50)
    print("  REAL-TIME LICENSE PLATE DETECTION")
    print("=" * 50)
    print("  Press 'Q' to quit")
    print("  Waiting for plate detection...")
    print("=" * 50)
    
    debug_log("Initializing camera for plate detection", "CAMERA")
    debug_log(f"Using camera device: {CAMERA_DEVICE}", "DEBUG")
    
    # Buka webcam dengan backend yang sesuai untuk OS
    if IS_WINDOWS:
        debug_log("Using DirectShow backend", "DEBUG")
        cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_DSHOW) if isinstance(CAMERA_DEVICE, int) else cv2.VideoCapture(CAMERA_DEVICE)
    elif IS_LINUX:
        debug_log("Using V4L2 backend", "DEBUG")
        cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2) if isinstance(CAMERA_DEVICE, int) else cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    else:
        debug_log("Using default backend", "DEBUG")
        cap = cv2.VideoCapture(CAMERA_DEVICE)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        debug_log("Camera initialization FAILED", "ERROR")
        print("   Try checking camera permissions or connection")
        if IS_LINUX:
            print("   On Linux, ensure user is in 'video' group: sudo usermod -a -G video $USER")
            print("   Check: ls -l /dev/video*")
        return None
    
    debug_log("Camera opened successfully", "SUCCESS")
    
    # Optimasi webcam settings berdasarkan performance mode
    debug_log(f"Applying {PERFORMANCE_MODE} performance settings", "PROCESS")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    debug_log(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps}fps", "DEBUG")
    debug_log(f"Performance: Process every {PROCESS_EVERY_N_FRAMES} frame(s), OCR scale: {OCR_SCALE}x", "INFO")
    
    # Detection stabilization: require plate detected in N consecutive frames
    stable_detection_count = 0
    
    detected_plate = None
    plate_detected_time = None
    is_counting_down = False
    countdown_start = 0
    captured_frame = None
    captured_crop = None
    frame_counter = 0
    process_counter = 0  # Counter untuk frame processing
    
    print("\n🎥 Starting webcam for plate detection...")
    print(f"⚡ Performance mode: {PERFORMANCE_MODE}")
    debug_log(f"Stabilization: plate must be detected in {STABLE_FRAMES_REQUIRED} consecutive frames", "INFO")
    debug_log("Entering detection loop", "PROCESS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            debug_log("Failed to read frame from camera", "ERROR")
            break
        
        frame_counter += 1
        process_counter += 1
        
        if DEBUG_MODE and frame_counter % 30 == 0:  # Log setiap 30 frame
            debug_log(f"Processing frame #{frame_counter}", "DEBUG")
        
        # Frame asli untuk deteksi (tanpa flip)
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Skip frame processing untuk performa (hanya proses setiap N frame)
        # Tapi tetap tampilkan display untuk smooth preview
        plate_found = False
        if process_counter >= PROCESS_EVERY_N_FRAMES:
            process_counter = 0
            # Deteksi plat dari frame asli
            plate_found, _, plate_crop, bbox = detect_plate_from_frame(model, frame, confidence)
            
            if DEBUG_MODE and plate_found and frame_counter % 10 == 0:
                debug_log(f"Plate detected at frame #{frame_counter}", "DEBUG")
        else:
            # Skip detection, assume no plate found
            bbox = None
        
        if plate_found and bbox:
            # Stabilization: increment counter
            stable_detection_count += 1
            
            x1, y1, x2, y2 = bbox
            
            # Gambar kotak deteksi
            color = (0, 255, 255) if stable_detection_count < STABLE_FRAMES_REQUIRED else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Show stabilization progress
            if stable_detection_count < STABLE_FRAMES_REQUIRED:
                cv2.putText(display_frame, f"DETECTING... {stable_detection_count}/{STABLE_FRAMES_REQUIRED}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "PLATE LOCKED", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mulai countdown hanya setelah stable detection
            if not is_counting_down and stable_detection_count >= STABLE_FRAMES_REQUIRED:
                is_counting_down = True
                countdown_start = time.time()
                print(f"\n🎯 Plate locked! Starting {CAPTURE_DELAY} second countdown...")
                debug_log(f"Countdown started at frame #{frame_counter} after {stable_detection_count} stable frames", "PROCESS")
            
            # Hitung sisa waktu countdown
            elapsed = time.time() - countdown_start
            remaining = CAPTURE_DELAY - elapsed
            
            if remaining > 0:
                # Tampilkan countdown
                countdown_text = f"Capturing in {remaining:.1f}s"
                cv2.putText(display_frame, countdown_text, (width//2 - 120, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Progress bar
                progress = elapsed / CAPTURE_DELAY
                bar_width = 300
                bar_x = width//2 - bar_width//2
                cv2.rectangle(display_frame, (bar_x, 70), (bar_x + bar_width, 90), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (bar_x, 70), (bar_x + int(bar_width * progress), 90), (0, 255, 0), -1)
            else:
                # Countdown selesai, capture gambar!
                captured_frame = frame.copy()
                captured_crop = plate_crop.copy()
                print(f"\n📸 Image captured! Processing OCR...")
                debug_log(f"Captured at frame #{frame_counter}, crop size: {plate_crop.shape}", "SUCCESS")
                break
        else:
            # Reset stabilization counter and countdown jika plat tidak terdeteksi
            if stable_detection_count > 0:
                debug_log(f"Plate lost after {stable_detection_count} frames", "WARNING")
            stable_detection_count = 0
            
            if is_counting_down:
                print("   ⚠️ Plate lost, resetting countdown...")
                debug_log("Countdown reset due to plate loss", "WARNING")
            is_counting_down = False
            countdown_start = 0
        
        # Tampilkan status
        if stable_detection_count > 0 and stable_detection_count < STABLE_FRAMES_REQUIRED:
            status_text = f"Stabilizing... ({stable_detection_count}/{STABLE_FRAMES_REQUIRED})"
        elif is_counting_down:
            status_text = "Hold still!"
        else:
            status_text = "Scanning for plates..."
        
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Press Q to quit", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame only if not in headless mode
        if not HEADLESS_MODE:
            cv2.imshow("License Plate Detection", display_frame)
        else:
            if DEBUG_MODE and frame_counter == 1:
                debug_log("Running in HEADLESS mode (no display)", "INFO")
        
        key = cv2.waitKey(1) & 0xFF if not HEADLESS_MODE else 0xFF
        if key == ord('q'):
            print("\n⏹ Detection cancelled by user.")
            break
    
    # Tutup webcam SEBELUM face recognition
    cap.release()
    cv2.destroyAllWindows()
    print("📷 Webcam closed.")
    
    # Proses OCR dari gambar yang sudah di-capture
    if captured_crop is not None:
        print("\n🔍 Processing captured plate image...")
        debug_log(f"Starting OCR on {captured_crop.shape} image", "PROCESS")
        
        # Save raw crop for debugging
        if DEBUG_MODE:
            cv2.imwrite("debug_raw_crop.jpg", captured_crop)
            debug_log("Raw crop saved to debug_raw_crop.jpg", "DEBUG")
        
        plate_text = process_plate_ocr(captured_crop)
        
        if plate_text:
            print(f"✅ DETECTED: {plate_text}")
            debug_log(f"OCR result: '{plate_text}'", "SUCCESS")
            save_plate_image(captured_crop, plate_text)
            debug_log(f"Plate image saved to {PLATES_FOLDER}", "SUCCESS")
            
            # Simpan frame dengan anotasi
            if captured_frame is not None:
                # Deteksi ulang untuk mendapatkan bbox
                _, _, _, bbox = detect_plate_from_frame(model, captured_frame, confidence)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(captured_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_bg_pt = (x1 + int(len(plate_text)*13), y1)
                    cv2.rectangle(captured_frame, (x1, y1-30), label_bg_pt, (0, 255, 0), -1)
                    cv2.putText(captured_frame, plate_text, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.imwrite(OUTPUT_PATH, captured_frame)
                print(f"   Output saved to {OUTPUT_PATH}")
            
            detected_plate = plate_text.replace(" ", "")
            return detected_plate
        else:
            print("❌ Could not read plate text from captured image.")
            return None
    else:
        print("No plate captured.")
    
    return None

def detect_license_plates(model, image_path, output_path, confidence=0.5):
    frame = cv2.imread(image_path)
    if frame is None: raise ValueError("Image not found")
    
    height, width = frame.shape[:2]
    results = model(frame, conf=confidence, verbose=False)
    
    print(f"Processing {image_path}...")
    
    detected_plate = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop dengan padding sedikit
            pad = 5
            crop = frame[max(0,y1-pad):min(height,y2+pad), max(0,x1-pad):min(width,x2+pad)]
            
            if crop.size > 0:
                text = read_plate_text(crop)
                
                if text:
                    print(f"✅ DETECTED: {text}")
                    save_plate_image(crop, text)
                    detected_plate = text.replace(" ", "")  # Simpan plat tanpa spasi
                    
                    # Visualisasi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_bg_pt = (x1 + int(len(text)*13), y1) 
                    cv2.rectangle(frame, (x1, y1-30), label_bg_pt, (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.imwrite(output_path, frame)
    print(f"Done! Output saved to {output_path}")
    
    if detected_plate:
        print("\n" + "=" * 50)
        print("  PLATE DETECTED - STARTING FACE VERIFICATION")
        print("=" * 50)
        print(f"  Plat terdeteksi: {detected_plate}")
        
        # Cek pemilik plat dari database
        owner = db.get_person_by_plate(detected_plate)
        if owner:
            print(f"  Pemilik terdaftar: {owner}")
        else:
            print(f"  ⚠️ Plat tidak terdaftar dalam database!")
        print("=" * 50)
        
        # Mulai face recognition dengan verifikasi plat
        face_recognizer.start_recognition_with_verification(detected_plate, db)
    else:
        print("No plate detected. Face recognition skipped.")

if __name__ == "__main__":
    MAX_VERIFICATION_ATTEMPTS = 2
    
    try:
        # Test camera connectivity if enabled
        if CAMERA_TEST_MODE:
            camera_ok = test_camera(CAMERA_DEVICE)
            if not camera_ok:
                print("\n⚠️ Camera test failed. Continue anyway? (y/n): ", end="")
                response = input().lower()
                if response != 'y':
                    print("❌ Exiting due to camera failure.")
                    exit(1)
        
        debug_log(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}", "INFO")
        debug_log(f"Headless mode: {'ENABLED' if HEADLESS_MODE else 'DISABLED'}", "INFO")
        debug_log(f"Camera device: {CAMERA_DEVICE}", "INFO")
        
        print("\n🔄 Loading YOLO model...")
        model = load_model(MODEL_PATH)
        debug_log(f"Model loaded from {MODEL_PATH}", "SUCCESS")
        
        print("\n" + "=" * 50)
        print("  LICENSE PLATE & FACE RECOGNITION SYSTEM")
        print("=" * 50)
        print("  System will continuously:")
        print("  1. Detect license plate")
        print("  2. Verify face (max 2 attempts)")
        print("  Press Ctrl+C to exit")
        print("=" * 50)
        
        while True:
            # ===== STEP 1: PLATE DETECTION =====
            print("\n" + "#" * 50)
            print("  STARTING PLATE DETECTION...")
            print("#" * 50)
            
            # Tampilkan di LCD
            if rasp_controller:
                rasp_controller.display_status("SCANNING...", "Show your plate")
            
            # Loop hingga plat terdaftar di database
            detected_plate = None
            owner = []  # List of owners
            
            while not owner:  # Loop terus sampai ada owner yang terdaftar
                detected_plate = detect_license_plates_realtime(model, CONFIDENCE_THRESHOLD)
                
                if detected_plate is None:
                    print("\n⚠️ No plate detected. Restarting detection...")
                    if rasp_controller:
                        rasp_controller.display_status("NO PLATE", "Try again...")
                    time.sleep(2)
                    continue
                
                # Cek pemilik plat dari database
                print(f"\n🔍 Checking plate: {detected_plate}")
                owner = db.get_person_by_plate(detected_plate)
                
                if not owner:  # Jika list kosong (tidak ada owner)
                    print(f"  ❌ Plat '{detected_plate}' TIDAK TERDAFTAR dalam database!")
                    print(f"  🔄 Mengulangi deteksi plat...")
                    print(f"  📝 Silakan scan plat yang terdaftar di sistem")
                    if rasp_controller:
                        rasp_controller.display_status("NOT REGISTERED", detected_plate)
                    time.sleep(3)
                    # Continue loop - akan detect lagi
                else:
                    print(f"  ✅ Plat terdaftar!")
                    print(f"  👤 Pemilik: {', '.join(owner)}")
                    if rasp_controller:
                        rasp_controller.display_status("PLATE OK", detected_plate)
                    time.sleep(1)
                    break
            
            # ===== STEP 2: FACE VERIFICATION =====
            print("\n" + "=" * 50)
            print("  PLATE DETECTED - STARTING FACE VERIFICATION")
            print("=" * 50)
            print(f"  Plat terdeteksi: {detected_plate}")
            print(f"  Pemilik terdaftar: {owner}")
            print("=" * 50)
            
            # Tampilkan di LCD
            if rasp_controller:
                owner_name = owner[0] if owner else "Unknown"
                rasp_controller.display_status("VERIFY FACE", owner_name[:16])
            
            # Delay sebentar sebelum memulai face recognition
            print("\n⏳ Preparing face recognition...")
            time.sleep(1)
            
            # Face verification dengan max 2 attempts (OTOMATIS)
            attempt = 0
            access_granted = False
            
            print("\n🤖 OTOMATIS: Memulai face recognition...")
            
            while attempt < MAX_VERIFICATION_ATTEMPTS and not access_granted:
                attempt += 1
                print(f"\n🔄 Verification attempt {attempt}/{MAX_VERIFICATION_ATTEMPTS}")
                
                # Mulai face recognition dengan verifikasi plat (webcam baru) - OTOMATIS
                result = face_recognizer.start_recognition_with_verification(
                    detected_plate, db, attempt, MAX_VERIFICATION_ATTEMPTS, auto_mode=True
                )
                
                if result == "GRANTED":
                    access_granted = True
                    print("\n✅ ACCESS GRANTED! Returning to plate detection...")
                    
                    # Trigger Raspberry Pi hardware - open gate
                    if rasp_controller:
                        print("🎛️ Triggering gate open sequence...")
                        rasp_controller.access_granted_sequence()
                    
                elif result == "DENIED":
                    if attempt < MAX_VERIFICATION_ATTEMPTS:
                        print(f"\n❌ ACCESS DENIED. {MAX_VERIFICATION_ATTEMPTS - attempt} attempt(s) remaining...")
                        time.sleep(2)
                    else:
                        print("\n❌ ACCESS DENIED. Maximum attempts reached.")
                        
                        # Trigger Raspberry Pi hardware - access denied sequence
                        if rasp_controller:
                            print("🎛️ Triggering access denied sequence...")
                            rasp_controller.access_denied_sequence()
                elif result == "CANCELLED":
                    print("\n⏹ Verification cancelled. Returning to plate detection...")
                    break
                elif result == "NO_FACE":
                    if attempt < MAX_VERIFICATION_ATTEMPTS:
                        print(f"\n⚠️ No face recognized. {MAX_VERIFICATION_ATTEMPTS - attempt} attempt(s) remaining...")
                        time.sleep(2)
                    else:
                        print("\n⚠️ No face recognized. Maximum attempts reached.")
            
            # Kembali ke plate detection
            print("\n" + "=" * 50)
            print("  RETURNING TO PLATE DETECTION...")
            print("=" * 50)
            if rasp_controller:
                rasp_controller.display_status("READY", "Waiting...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n⏹ System stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup Raspberry Pi Controller
        if rasp_controller:
            print("\n🎛️ Cleaning up Raspberry Pi hardware...")
            rasp_controller.cleanup()
        
        db.close()
        print("\n👋 Goodbye!")
