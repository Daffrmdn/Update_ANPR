"""
Modul Face Recognition menggunakan DeepFace
Terintegrasi dengan sistem deteksi plat nomor
OPTIMIZED FOR HIGH FPS
"""

import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import threading
import time
import platform
from queue import Queue
from collections import deque

# Fix Qt platform plugin warnings for Raspberry Pi
if platform.system() == 'Linux':
    os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 instead of Wayland
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'  # Suppress Qt warnings

# Deteksi sistem operasi
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_RASPBERRY_PI = IS_LINUX and os.path.exists('/proc/device-tree/model')

# Konfigurasi
FACES_DATABASE = "faces_database"
EMBEDDINGS_FILE = "face_embeddings.pkl"

# Camera config
CAMERA_DEVICE = "/dev/video0" if IS_LINUX else 0  # Auto-select based on OS

# Debug mode
DEBUG_MODE = False  # Set False untuk production
HEADLESS_MODE = os.environ.get('DISPLAY') is None if IS_LINUX else False
AUTO_GRANT_FACE = True  # Set True untuk auto-grant ketika wajah terdeteksi (debugging)

# ============ OPTIMASI FPS ============
# Performance mode untuk Raspberry Pi
PERFORMANCE_MODE = "LOW" if IS_RASPBERRY_PI else "BALANCED"

# Pilih model (dari tercepat ke paling akurat):
# - "SFace" : Tercepat, akurasi cukup
# - "Facenet" : Cepat, akurasi bagus (RECOMMENDED - balance)
# - "Facenet512" : Lebih lambat, akurasi tinggi (BEST ACCURACY)
# - "ArcFace" : Lambat, akurasi sangat tinggi
# PENTING: Model HARUS SAMA dengan yang dipakai saat training!
# Gunakan model lebih ringan di Raspberry Pi untuk performa
if PERFORMANCE_MODE == "LOW":
    MODEL_NAME = "Facenet"  # Lebih cepat untuk Raspberry Pi
    DETECTOR_BACKEND = "opencv"  # Tercepat
    PROCESS_SCALE = 0.5
    SKIP_FRAMES = 3
elif PERFORMANCE_MODE == "BALANCED":
    MODEL_NAME = "Facenet512"
    DETECTOR_BACKEND = "ssd"
    PROCESS_SCALE = 0.75
    SKIP_FRAMES = 2
else:  # HIGH
    MODEL_NAME = "Facenet512"
    DETECTOR_BACKEND = "ssd"
    PROCESS_SCALE = 1.0
    SKIP_FRAMES = 1

# Detector backend (dari tercepat ke paling akurat):
# - "opencv" : Tercepat
# - "ssd" : Cepat, akurasi bagus
# - "mtcnn" : Akurat tapi lambat
# - "retinaface" : Paling akurat tapi paling lambat

THRESHOLD = 0.50  # Threshold similarity (0.50 = balance, naikkan untuk lebih ketat)
RECOGNITION_LOG = "recognition_log"

# ===== DEBUG FUNCTION =====
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
        "PROCESS": "⚙️",
        "FACE": "👤"
    }
    icon = icons.get(level, "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {message}")

def test_camera_readiness(camera_device, backend=None, max_test_frames=5):
    """Test if camera is ready and can capture frames"""
    print("\n" + "=" * 50)
    print("  🔍 TESTING CAMERA READINESS")
    print("=" * 50)
    print(f"📷 Camera device: {camera_device}")
    print(f"🔧 Backend: {backend if backend else 'default'}")
    print(f"🧪 Test frames: {max_test_frames}")
    print("=" * 50)
    
    cap = None
    try:
        # Open camera
        if backend:
            cap = cv2.VideoCapture(camera_device, backend)
        else:
            cap = cv2.VideoCapture(camera_device)
        
        if not cap.isOpened():
            print("❌ Camera failed to open")
            return False, None
        
        print("✅ Camera opened successfully")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📐 Resolution: {width}x{height}")
        print(f"🎞️  FPS: {fps}")
        
        # Test frame capture
        print(f"\n🧪 Testing frame capture ({max_test_frames} frames)...")
        successful_frames = 0
        
        for i in range(max_test_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                successful_frames += 1
                print(f"   Frame {i+1}/{max_test_frames}: ✅ OK ({frame.shape})")
            else:
                print(f"   Frame {i+1}/{max_test_frames}: ❌ Failed")
            time.sleep(0.1)  # Small delay between frames
        
        success_rate = (successful_frames / max_test_frames) * 100
        print(f"\n📊 Success rate: {successful_frames}/{max_test_frames} ({success_rate:.0f}%)")
        
        if successful_frames >= max_test_frames * 0.8:  # 80% success rate
            print("✅ Camera is READY")
            print("=" * 50)
            return True, cap
        else:
            print("❌ Camera NOT ready (too many failed frames)")
            print("=" * 50)
            if cap:
                cap.release()
            return False, None
            
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        print("=" * 50)
        if cap:
            cap.release()
        return False, None

class FaceRecognizer:
    def __init__(self):
        """Inisialisasi Face Recognizer"""
        self.embeddings_db = None
        self.model = None
        self.is_running = False
        self.deepface_available = False
        
        # Threading untuk async processing
        self.process_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.last_results = []  # Cache hasil terakhir
        self.last_boxes = []    # Cache bounding boxes
        
        # Cek DeepFace
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
            print("   ✅ DeepFace module loaded")
            print(f"   📊 Using model: {MODEL_NAME}, detector: {DETECTOR_BACKEND}")
        except ImportError:
            print("   ⚠️ DeepFace not installed. Face recognition disabled.")
            print("      Run: pip install deepface tensorflow")
            return
        
        # Pre-load model untuk menghindari delay saat pertama kali
        self._preload_model()
        
        # Load embeddings database
        self._load_embeddings()
        
        # Buat folder log
        if not os.path.exists(RECOGNITION_LOG):
            os.makedirs(RECOGNITION_LOG)
    
    def _preload_model(self):
        """Pre-load model agar tidak delay saat pertama kali"""
        try:
            print("   🔄 Pre-loading face recognition model...")
            # Buat dummy image untuk trigger model loading
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            self.DeepFace.represent(
                img_path=dummy,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )
            print("   ✅ Model pre-loaded")
        except:
            pass
    
    def _load_embeddings(self):
        """Load database embeddings dari file"""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                num_persons = len(self.embeddings_db.get('persons', {}))
                print(f"   ✅ Face database loaded: {num_persons} persons")
            except Exception as e:
                print(f"   ⚠️ Error loading embeddings: {e}")
                self.embeddings_db = None
        else:
            print(f"   ⚠️ No face database found ({EMBEDDINGS_FILE})")
            print("      Run 'python train_faces.py' to create database")
    
    def _calculate_similarity(self, emb1, emb2):
        """Menghitung cosine similarity antara dua embedding"""
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def _find_best_match(self, face_embedding):
        """Mencari kecocokan terbaik dari database"""
        if not self.embeddings_db or 'persons' not in self.embeddings_db:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for person_name, embeddings_list in self.embeddings_db['persons'].items():
            for emb_data in embeddings_list:
                stored_embedding = emb_data['embedding']
                similarity = self._calculate_similarity(face_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
        
        return best_match, best_similarity
    
    def recognize_face(self, frame, use_scale=True):
        """Mengenali wajah dalam frame (OPTIMIZED)"""
        if not self.deepface_available:
            return frame, []
        
        results = []
        display_frame = frame.copy()
        
        # Resize frame untuk proses lebih cepat
        if use_scale and PROCESS_SCALE < 1.0:
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (int(w * PROCESS_SCALE), int(h * PROCESS_SCALE)))
            scale_factor = 1.0 / PROCESS_SCALE
        else:
            small_frame = frame
            scale_factor = 1.0
        
        try:
            # Ekstrak wajah dan embedding menggunakan DeepFace
            faces = self.DeepFace.represent(
                img_path=small_frame,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )
            
            for face_data in faces:
                embedding = face_data['embedding']
                facial_area = face_data.get('facial_area', {})
                
                # Cari kecocokan
                name, similarity = self._find_best_match(embedding)
                
                # Tentukan hasil
                if similarity >= THRESHOLD:
                    label = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0)  # Hijau untuk dikenali
                    recognized = True
                else:
                    label = f"Unknown ({similarity:.2f})"
                    color = (0, 0, 255)  # Merah untuk tidak dikenali
                    name = "Unknown"
                    recognized = False
                
                # Gambar kotak wajah (scale back ke ukuran asli)
                if facial_area:
                    x = int(facial_area.get('x', 0) * scale_factor)
                    y = int(facial_area.get('y', 0) * scale_factor)
                    w = int(facial_area.get('w', 100) * scale_factor)
                    h = int(facial_area.get('h', 100) * scale_factor)
                    
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Background untuk label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x, y-30), (x+label_size[0]+10, y), color, -1)
                    cv2.putText(display_frame, label, (x+5, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Simpan bbox untuk cache
                    facial_area = {'x': x, 'y': y, 'w': w, 'h': h}
                
                results.append({
                    'name': name,
                    'similarity': similarity,
                    'recognized': recognized,
                    'bbox': facial_area,
                    'label': label,
                    'color': color
                })
                
        except Exception as e:
            # Tidak ada wajah terdeteksi atau error lainnya
            pass
        
        # Update cache
        self.last_results = results
        
        return display_frame, results
    
    def _draw_cached_results(self, frame):
        """Gambar hasil cached ke frame (untuk frame yang di-skip)"""
        display_frame = frame.copy()
        
        for r in self.last_results:
            bbox = r.get('bbox', {})
            if bbox:
                x = bbox.get('x', 0)
                y = bbox.get('y', 0)
                w = bbox.get('w', 100)
                h = bbox.get('h', 100)
                color = r.get('color', (0, 255, 0))
                label = r.get('label', '')
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x, y-30), (x+label_size[0]+10, y), color, -1)
                cv2.putText(display_frame, label, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def start_recognition(self):
        """Memulai face recognition real-time dengan webcam (OPTIMIZED FOR HIGH FPS)"""
        if not self.deepface_available:
            print("❌ DeepFace not available. Cannot start recognition.")
            return
        
        if not self.embeddings_db:
            print("❌ No face database loaded. Run training first!")
            return
        
        print("\n" + "=" * 50)
        print("  FACE RECOGNITION - REAL TIME (OPTIMIZED)")
        print("=" * 50)
        print(f"  Model: {MODEL_NAME}")
        print(f"  Detector: {DETECTOR_BACKEND}")
        print(f"  Process Scale: {PROCESS_SCALE}")
        print(f"  Skip Frames: {SKIP_FRAMES}")
        print("-" * 50)
        print("  Press 'Q' to quit")
        print("  Press 'S' to save screenshot")
        print("  Press '+'/'-' to adjust skip frames")
        print("-" * 50)
        
        debug_log("Initializing camera for face recognition", "CAMERA")
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
            if IS_LINUX:
                print("   On Linux, ensure user is in 'video' group: sudo usermod -a -G video $USER")
            return
        
        debug_log("Camera opened successfully", "SUCCESS")
        
        # Optimasi webcam settings berdasarkan performance mode
        if PERFORMANCE_MODE == "LOW":
            cam_width, cam_height, cam_fps = 640, 480, 10
        elif PERFORMANCE_MODE == "BALANCED":
            cam_width, cam_height, cam_fps = 640, 480, 20
        else:  # HIGH
            cam_width, cam_height, cam_fps = 1280, 720, 30
        
        debug_log(f"Applying {PERFORMANCE_MODE} performance settings", "PROCESS")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer untuk latency rendah
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        debug_log(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps}fps", "DEBUG")
        
        self.is_running = True
        recognition_history = []
        
        # Stats
        frame_count = 0
        total_frames = 0
        start_time = time.time()
        current_fps = 0
        skip_counter = 0
        current_skip = SKIP_FRAMES
        
        # Untuk smooth FPS calculation
        fps_history = deque(maxlen=30)
        last_frame_time = time.time()
        
        print("\n🎥 Starting webcam...")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            total_frames += 1
            current_time = time.time()
            
            # Hitung FPS (smoothed)
            frame_time = current_time - last_frame_time
            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
            last_frame_time = current_time
            
            if len(fps_history) > 0:
                current_fps = sum(fps_history) / len(fps_history)
            
            # Skip frame logic - proses hanya setiap N frame
            skip_counter += 1
            
            if skip_counter >= current_skip:
                # Proses frame ini
                skip_counter = 0
                display_frame, results = self.recognize_face(frame)
                
                # Log hasil
                for r in results:
                    if r['recognized']:
                        log_entry = {
                            'time': datetime.now().isoformat(),
                            'name': r['name'],
                            'similarity': r['similarity']
                        }
                        recognition_history.append(log_entry)
                        print(f"   👤 Recognized: {r['name']} (similarity: {r['similarity']:.2f})")
            else:
                # Gunakan cached results untuk frame yang di-skip
                display_frame = self._draw_cached_results(frame)
            
            # Tampilkan info
            info_color = (0, 255, 255)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
            cv2.putText(display_frame, f"Skip: {current_skip} (+/-)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"DB: {len(self.embeddings_db.get('persons', {}))} persons", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Tampilkan recognition history
            y_offset = 120
            for entry in recognition_history[-3:]:
                text = f"• {entry['name']}"
                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                y_offset += 20
            
            # Show frame only if not in headless mode
            if not HEADLESS_MODE:
                cv2.imshow("Face Recognition (Q=quit, +/-=adjust)", display_frame)
            
            key = cv2.waitKey(1) & 0xFF if not HEADLESS_MODE else 0xFF
            
            if key == ord('q'):
                print("\n⏹ Stopping recognition...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(RECOGNITION_LOG, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"   📸 Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                current_skip = min(current_skip + 1, 10)
                print(f"   ⬆️ Skip frames: {current_skip}")
            elif key == ord('-'):
                current_skip = max(current_skip - 1, 1)
                print(f"   ⬇️ Skip frames: {current_skip}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Simpan log
        if recognition_history:
            log_file = os.path.join(RECOGNITION_LOG, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(log_file, 'w') as f:
                for entry in recognition_history:
                    f.write(f"{entry['time']},{entry['name']},{entry['similarity']:.4f}\n")
            print(f"\n📝 Log saved: {log_file}")
        
        print("✅ Face recognition stopped.")
    
    def start_recognition_with_verification(self, detected_plate, db, attempt=1, max_attempts=2, auto_mode=False):
        """
        Face recognition dengan verifikasi plat nomor
        Mengecek apakah wajah yang terdeteksi cocok dengan pemilik plat
        
        Args:
            detected_plate: Nomor plat yang terdeteksi
            db: Database instance
            attempt: Percobaan ke-berapa
            max_attempts: Maksimal percobaan
            auto_mode: Jika True, akan otomatis memverifikasi tanpa tombol 'V'
        
        Returns: "GRANTED", "DENIED", "CANCELLED", "NO_FACE"
        """
        if not self.deepface_available:
            print("❌ DeepFace not available. Cannot start recognition.")
            return "CANCELLED"
        
        if not self.embeddings_db:
            print("❌ No face database loaded. Run training first!")
            return "CANCELLED"
        
        # Cek pemilik plat dari database
        expected_owner = db.get_person_by_plate(detected_plate)
        
        print("\n" + "=" * 50)
        mode_text = "OTOMATIS" if auto_mode else "MANUAL"
        print(f"  FACE VERIFICATION MODE [{mode_text}] (Attempt {attempt}/{max_attempts})")
        print("=" * 50)
        print(f"  Plat: {detected_plate}")
        if expected_owner:
            print(f"  Pemilik terdaftar: {expected_owner}")
        else:
            print("  ⚠️ PLAT TIDAK TERDAFTAR - Mode pengawasan")
        print("-" * 50)
        if auto_mode:
            print("  🤖 Mode OTOMATIS - akan verify otomatis saat wajah terdeteksi")
            print("  Press 'Q' to quit/cancel")
        else:
            print("  Press 'Q' to quit/cancel")
            print("  Press 'V' to verify (capture & check)")
        print("=" * 50)
        
        debug_log("Initializing camera for face verification", "CAMERA")
        debug_log(f"Using camera device: {CAMERA_DEVICE}", "DEBUG")
        
        # Release any lingering camera resources
        cv2.destroyAllWindows()
        print("\n⏳ Releasing previous camera resources...")
        time.sleep(0.3)  # Delay to ensure resources are released
        
        # Additional delay for Raspberry Pi
        if IS_RASPBERRY_PI:
            print("🍓 Raspberry Pi detected - waiting for camera to be ready...")
            time.sleep(1.0)  # Extra delay for Pi
        
        # Determine backend
        backend = None
        backend_name = "default"
        if IS_WINDOWS:
            backend = cv2.CAP_DSHOW
            backend_name = "DirectShow"
        elif IS_LINUX:
            backend = cv2.CAP_V4L2
            backend_name = "V4L2"
        
        # Buka webcam dengan backend yang sesuai untuk OS - with retry logic
        max_retries = 3
        cap = None
        camera_ready = False
        
        for retry in range(max_retries):
            print(f"\n🔄 Attempt {retry + 1}/{max_retries} - Testing camera readiness...")
            
            try:
                # Test camera readiness with frame capture test
                camera_ready, cap = test_camera_readiness(
                    CAMERA_DEVICE, 
                    backend,
                    max_test_frames=5
                )
                
                if camera_ready and cap and cap.isOpened():
                    print(f"\n✅ Camera is READY on attempt {retry + 1}!")
                    break
                else:
                    if retry < max_retries - 1:
                        print(f"\n⚠️ Camera not ready, waiting and retrying...")
                        if cap:
                            cap.release()
                        time.sleep(1.5)  # Wait before retry
                    else:
                        print(f"\n❌ Cannot initialize camera after {max_retries} attempts!")
                        if IS_LINUX:
                            print("   💡 Tips for Linux/Raspberry Pi:")
                            print("   1. Check camera connection: vcgencmd get_camera")
                            print("   2. Add user to video group: sudo usermod -a -G video $USER")
                            print("   3. Check if camera is in use: sudo fuser /dev/video0")
                            print("   4. Reboot if needed: sudo reboot")
                        return "CANCELLED"
            except Exception as e:
                print(f"❌ Camera initialization error: {e}")
                if retry < max_retries - 1:
                    print("⏳ Waiting before retry...")
                    time.sleep(1.5)
                else:
                    return "CANCELLED"
        
        if not camera_ready or not cap or not cap.isOpened():
            print("\n❌ Cannot open webcam!")
            return "CANCELLED"
        
        # Optimasi webcam settings berdasarkan performance mode
        if PERFORMANCE_MODE == "LOW":
            cam_width, cam_height, cam_fps = 640, 480, 10
        elif PERFORMANCE_MODE == "BALANCED":
            cam_width, cam_height, cam_fps = 640, 480, 20
        else:  # HIGH
            cam_width, cam_height, cam_fps = 1280, 720, 30
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        verification_result = None  # Will be "GRANTED", "DENIED", "CANCELLED", "NO_FACE"
        
        # Auto mode variables
        auto_verify_timer = None
        auto_verify_delay = 2.0  # Delay 2 detik setelah wajah terdeteksi dalam mode auto
        stable_face_count = 0  # Counter untuk face yang stabil
        required_stable_frames = 10  # Butuh 10 frame stabil sebelum auto-verify
        last_detected_name = None
        
        # Stats
        frame_count = 0
        skip_counter = 0
        current_skip = SKIP_FRAMES
        fps_history = deque(maxlen=30)
        last_frame_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            
            # Calculate FPS
            frame_time = current_time - last_frame_time
            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
            last_frame_time = current_time
            current_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Skip frame logic
            skip_counter += 1
            if skip_counter >= current_skip:
                skip_counter = 0
                display_frame, results = self.recognize_face(frame)
            else:
                display_frame = self._draw_cached_results(frame)
                results = self.last_results
            
            # Draw info panel
            panel_color = (50, 50, 50)
            cv2.rectangle(display_frame, (0, 0), (350, 160), panel_color, -1)
            
            # Attempt info
            cv2.putText(display_frame, f"Attempt: {attempt}/{max_attempts}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            
            # Info text
            cv2.putText(display_frame, f"PLAT: {detected_plate}", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            owner_text = f"Owner: {expected_owner}" if expected_owner else "Owner: NOT REGISTERED"
            owner_color = (0, 255, 0) if expected_owner else (0, 0, 255)
            cv2.putText(display_frame, owner_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, owner_color, 1)
            
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show detected face
            if results:
                detected_name = results[0]['name']
                similarity = results[0]['similarity']
                
                face_text = f"Face: {detected_name} ({similarity:.2f})"
                face_color = (0, 255, 0) if results[0]['recognized'] else (0, 0, 255)
                cv2.putText(display_frame, face_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
                
                # Auto verification check
                # Handle expected_owner as list or string
                is_match = False
                if expected_owner:
                    if isinstance(expected_owner, list):
                        # Check if detected name is in the list of owners
                        is_match = detected_name.upper() in [owner.upper() for owner in expected_owner]
                    else:
                        # Single owner (string)
                        is_match = detected_name.upper() == expected_owner.upper()
                
                if expected_owner and is_match:
                    status_text = "STATUS: MATCH ✓"
                    status_color = (0, 255, 0)
                elif expected_owner:
                    status_text = "STATUS: MISMATCH ✗"
                    status_color = (0, 0, 255)
                else:
                    status_text = "STATUS: UNREGISTERED PLATE"
                    status_color = (0, 165, 255)
                
                cv2.putText(display_frame, status_text, (10, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # AUTO MODE LOGIC
                if auto_mode and results[0]['recognized']:
                    # Cek apakah wajah yang terdeteksi sama dengan frame sebelumnya (stabilisasi)
                    if detected_name == last_detected_name:
                        stable_face_count += 1
                    else:
                        stable_face_count = 1
                        last_detected_name = detected_name
                        auto_verify_timer = None
                    
                    # Jika wajah stabil, mulai countdown
                    if stable_face_count >= required_stable_frames:
                        if auto_verify_timer is None:
                            auto_verify_timer = time.time()
                            print(f"\n✅ Wajah terdeteksi stabil: {detected_name}")
                            print(f"⏱️ Memulai countdown {auto_verify_delay} detik untuk auto-verify...")
                        
                        elapsed = time.time() - auto_verify_timer
                        remaining = auto_verify_delay - elapsed
                        
                        if remaining > 0:
                            # Tampilkan countdown
                            countdown_text = f"AUTO-VERIFY in {remaining:.1f}s"
                            cv2.putText(display_frame, countdown_text, 
                                       (display_frame.shape[1]//2 - 150, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        # Countdown logic akan dihandle di bawah setelah display frame
                else:
                    # Reset auto-verify timer jika tidak ada wajah atau mode manual
                    if stable_face_count > 0:
                        stable_face_count = 0
                        last_detected_name = None
                        auto_verify_timer = None
            else:
                cv2.putText(display_frame, "Face: Scanning...", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                # Reset auto-verify jika tidak ada wajah terdeteksi
                stable_face_count = 0
                last_detected_name = None
                auto_verify_timer = None
            
            # Instructions
            if auto_mode:
                cv2.putText(display_frame, "AUTO MODE - Press Q to quit", 
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(display_frame, "Press V to verify, Q to quit", 
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame only if not in headless mode
            if not HEADLESS_MODE:
                cv2.imshow("Face Verification", display_frame)
            
            # Check for auto-verify trigger BEFORE waiting for key
            trigger_verify = False
            
            # Auto-verify logic (cek apakah countdown selesai)
            if auto_mode and auto_verify_timer is not None:
                elapsed = time.time() - auto_verify_timer
                if elapsed >= auto_verify_delay:
                    trigger_verify = True
                    print(f"\n🤖 AUTO-VERIFY triggered untuk: {last_detected_name}")
            
            key = cv2.waitKey(1) & 0xFF if not HEADLESS_MODE else 0xFF
            
            if key == ord('q'):
                print("\n⏹ Verification cancelled.")
                verification_result = "CANCELLED"
                break
            
            elif key == ord('v') or trigger_verify:
                # Manual verification (V) atau Auto-verify trigger
                if results and results[0]['recognized']:
                    detected_name = results[0]['name']
                    
                    # AUTO GRANT MODE - jika AUTO_GRANT_FACE aktif, auto-grant setiap wajah terdeteksi
                    if AUTO_GRANT_FACE:
                        is_valid = True
                        message = f"🔓 [DEBUG MODE] Auto-granted for detected face: {detected_name}"
                        
                        print(f"\n{'=' * 50}")
                        print(f"  VERIFICATION RESULT (AUTO GRANT MODE)")
                        print(f"{'=' * 50}")
                        print(f"  Plat: {detected_plate}")
                        print(f"  Wajah terdeteksi: {detected_name}")
                        print(f"  {message}")
                        print(f"  ⚠️ AUTO_GRANT_FACE is enabled (debugging)")
                        print(f"{'=' * 50}")
                    else:
                        # Normal verification
                        is_valid, message = db.verify_access(detected_name, detected_plate)
                        
                        print(f"\n{'=' * 50}")
                        print(f"  VERIFICATION RESULT")
                        print(f"{'=' * 50}")
                        print(f"  Plat: {detected_plate}")
                        print(f"  Wajah terdeteksi: {detected_name}")
                        print(f"  {message}")
                        print(f"{'=' * 50}")
                    
                    # Log ke database
                    status = "VALID" if is_valid else "DENIED"
                    db.log_access(detected_name, detected_plate, status)
                    
                    verification_result = "GRANTED" if is_valid else "DENIED"
                    
                    # Show result on screen
                    result_color = (0, 255, 0) if is_valid else (0, 0, 255)
                    result_text = "ACCESS GRANTED" if is_valid else "ACCESS DENIED"
                    
                    # Create result overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (100, 180), (540, 300), result_color, -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    cv2.putText(display_frame, result_text, (140, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    
                    if not HEADLESS_MODE:
                        cv2.imshow("Face Verification", display_frame)
                        cv2.waitKey(2000)  # Show result for 2 seconds
                    else:
                        time.sleep(2)  # Wait without display
                    break
                    
                else:
                    # AUTO GRANT MODE - jika ada wajah apapun terdeteksi (meski unknown), auto-grant
                    if AUTO_GRANT_FACE and results:
                        # Ada wajah terdeteksi tapi tidak recognized (Unknown)
                        detected_name = results[0]['name']  # "Unknown"
                        
                        print(f"\n{'=' * 50}")
                        print(f"  VERIFICATION RESULT (AUTO GRANT MODE)")
                        print(f"{'=' * 50}")
                        print(f"  Plat: {detected_plate}")
                        print(f"  Wajah terdeteksi: {detected_name} (Unknown)")
                        print(f"  🔓 [DEBUG MODE] Auto-granted for unknown face")
                        print(f"  ⚠️ AUTO_GRANT_FACE is enabled (debugging)")
                        print(f"{'=' * 50}")
                        
                        # Log ke database
                        db.log_access(detected_name, detected_plate, "VALID_AUTO_GRANT")
                        
                        verification_result = "GRANTED"
                        
                        # Show result on screen
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (100, 180), (540, 300), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                        cv2.putText(display_frame, "ACCESS GRANTED", (140, 250),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        
                        if not HEADLESS_MODE:
                            cv2.imshow("Face Verification", display_frame)
                            cv2.waitKey(2000)
                        else:
                            time.sleep(2)
                        break
                    else:
                        # Mode normal - tidak ada wajah terdeteksi
                        print("\n⚠️ Tidak ada wajah yang dikenali. Coba lagi.")
                        verification_result = "NO_FACE"
                        
                        # Show warning on screen
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (100, 180), (540, 300), (0, 165, 255), -1)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                        cv2.putText(display_frame, "NO FACE DETECTED", (120, 250),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        
                        if not HEADLESS_MODE:
                            cv2.imshow("Face Verification", display_frame)
                            cv2.waitKey(1500)
                        else:
                            time.sleep(1.5)  # Wait without display
                        break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if verification_result is None:
            verification_result = "CANCELLED"
        
        return verification_result
    
    def stop_recognition(self):
        """Menghentikan recognition"""
        self.is_running = False
    
    def get_persons(self):
        """Mendapatkan daftar orang dalam database"""
        if not self.embeddings_db or 'persons' not in self.embeddings_db:
            return []
        return list(self.embeddings_db['persons'].keys())
    
    def reload_database(self):
        """Reload database embeddings"""
        self._load_embeddings()
        print("✅ Database reloaded")


# Test module
if __name__ == "__main__":
    print("Testing Face Recognition Module...")
    
    recognizer = FaceRecognizer()
    
    if recognizer.deepface_available:
        persons = recognizer.get_persons()
        print(f"\nPersons in database: {persons}")
        
        # Start recognition
        print("\nStarting real-time recognition...")
        recognizer.start_recognition()
    else:
        print("\nPlease install DeepFace: pip install deepface tensorflow")
