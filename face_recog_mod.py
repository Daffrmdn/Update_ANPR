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
AUTO_GRANT_FACE = False  # Set False untuk production (enable real face verification)
TESTING_MODE = False  # Set True untuk bypass face verification (testing only)

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
    DETECTOR_BACKEND = "ssd"  # Lebih akurat dari opencv (worth the trade-off)
    PROCESS_SCALE = 0.75  # Naikkan dari 0.5 ke 0.75 untuk better detection
    SKIP_FRAMES = 3
elif PERFORMANCE_MODE == "BALANCED":
    MODEL_NAME = "Facenet512"
    DETECTOR_BACKEND = "ssd"
    PROCESS_SCALE = 0.85  # Sedikit naikkan
    SKIP_FRAMES = 2
else:  # HIGH
    MODEL_NAME = "Facenet512"
    DETECTOR_BACKEND = "ssd"
    PROCESS_SCALE = 1.0
    SKIP_FRAMES = 1

# Detector backend (dari tercepat ke paling akurat):
# - "opencv" : Tercepat (kurang akurat)
# - "ssd" : Cepat, akurasi bagus (RECOMMENDED)
# - "mtcnn" : Akurat tapi lambat
# - "retinaface" : Paling akurat tapi paling lambat

THRESHOLD = 0.50  # Threshold similarity (0.50 = balance, naikkan untuk lebih ketat)
EDGE_CONFIDENCE_THRESHOLD = 0.4  # Lower threshold untuk edge cases

# Image enhancement untuk better face detection
ENHANCE_BRIGHTNESS = True  # Auto-enhance brightness untuk foto gelap
ENHANCE_CONTRAST = True    # CLAHE contrast enhancement
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
    
    def _enhance_image(self, image):
        """Enhance image untuk face detection yang lebih baik"""
        enhanced = image.copy()
        
        # Brightness enhancement jika terlalu gelap
        if ENHANCE_BRIGHTNESS:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 100:  # Terlalu gelap
                # Brightness adjustment
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                # Increase brightness
                value = int((100 - avg_brightness) * 0.6)  # 60% dari deficit
                v = cv2.add(v, value)
                
                hsv = cv2.merge([h, s, v])
                enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                debug_log(f"Brightness enhanced: {avg_brightness:.0f} -> ~{avg_brightness + value:.0f}", "PROCESS")
        
        # Contrast enhancement
        if ENHANCE_CONTRAST:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            debug_log("Contrast enhanced with CLAHE", "PROCESS")
        
        return enhanced
    
    def recognize_face(self, frame, use_scale=True):
        """Mengenali wajah dalam frame (OPTIMIZED)"""
        if not self.deepface_available:
            return frame, []
        
        results = []
        display_frame = frame.copy()
        
        # Enhance image sebelum processing
        enhanced_frame = self._enhance_image(frame)
        
        # Resize frame untuk proses lebih cepat
        if use_scale and PROCESS_SCALE < 1.0:
            h, w = enhanced_frame.shape[:2]
            small_frame = cv2.resize(enhanced_frame, (int(w * PROCESS_SCALE), int(h * PROCESS_SCALE)))
            scale_factor = 1.0 / PROCESS_SCALE
        else:
            small_frame = enhanced_frame
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
        SIMPLIFIED: Countdown 5 detik → Capture → Analyze
        
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
        print(f"  FACE VERIFICATION - SIMPLIFIED MODE")
        print(f"  (Attempt {attempt}/{max_attempts})")
        print("=" * 50)
        print(f"  Plat: {detected_plate}")
        if expected_owner:
            print(f"  Pemilik terdaftar: {expected_owner}")
        else:
            print("  ⚠️ PLAT TIDAK TERDAFTAR - Mode pengawasan")
        print("-" * 50)
        print("  📸 System will capture in 5 seconds...")
        print("  Press 'Q' to cancel")
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
        
        # Simple camera open (no complex retry for simplified mode)
        print("\n📷 Opening camera...")
        if backend:
            cap = cv2.VideoCapture(CAMERA_DEVICE, backend)
        else:
            cap = cv2.VideoCapture(CAMERA_DEVICE)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            if IS_LINUX:
                print("   💡 Tips for Linux/Raspberry Pi:")
                print("   1. Check camera connection: vcgencmd get_camera")
                print("   2. Add user to video group: sudo usermod -a -G video $USER")
                print("   3. Check if camera is in use: sudo fuser /dev/video0")
            return "CANCELLED"
        
        print("✅ Camera opened successfully!")
        
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
        
        # ===== NEW SIMPLIFIED LOGIC: 5 SECOND COUNTDOWN → CAPTURE → ANALYZE =====
        COUNTDOWN_DURATION = 5.0  # 5 detik
        countdown_start = time.time()
        captured_frame = None
        
        print("\n🎬 Starting 5 second countdown...")
        print("   Position your face in front of the camera")
        
        # Countdown loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Calculate remaining time
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_DURATION - elapsed
            
            # Draw countdown overlay
            height, width = display_frame.shape[:2]
            
            # Semi-transparent overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 150), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Info text
            cv2.putText(display_frame, f"PLAT: {detected_plate}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            owner_text = f"Owner: {', '.join(expected_owner) if isinstance(expected_owner, list) else expected_owner}" if expected_owner else "Owner: UNREGISTERED"
            cv2.putText(display_frame, owner_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Quick face detection preview (non-blocking)
            try:
                faces_preview = self.DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                face_count = len(faces_preview) if faces_preview else 0
            except:
                face_count = 0
            
            # Countdown display
            if remaining > 0:
                countdown_text = f"Capturing in: {remaining:.1f}s"
                cv2.putText(display_frame, countdown_text, (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Show face detection status
                face_status = f"Face detected: {face_count}" if face_count > 0 else "No face detected yet"
                face_color = (0, 255, 0) if face_count > 0 else (0, 165, 255)
                cv2.putText(display_frame, face_status, (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
                
                # Progress bar
                bar_width = width - 40
                bar_x = 20
                bar_y = height - 40
                progress = elapsed / COUNTDOWN_DURATION
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 20), (0, 255, 0), -1)
            else:
                # Countdown complete - capture!
                captured_frame = frame.copy()
                cv2.putText(display_frame, "CAPTURING...", (width//2 - 100, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                if not HEADLESS_MODE:
                    cv2.imshow("Face Verification", display_frame)
                    cv2.waitKey(500)  # Show "CAPTURING" for 0.5 sec
                else:
                    time.sleep(0.5)
                
                print("\n📸 Frame captured! Analyzing...")
                break
            
            # Show frame
            if not HEADLESS_MODE:
                cv2.imshow("Face Verification", display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF if not HEADLESS_MODE else 0xFF
            if key == ord('q'):
                print("\n⏹ Verification cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return "CANCELLED"
        
        # Close camera
        cap.release()
        cv2.destroyAllWindows()
        
        # ===== ANALYZE CAPTURED FRAME =====
        if captured_frame is None:
            print("❌ No frame captured!")
            return "CANCELLED"
        
        print("\n🔍 Analyzing captured image for face recognition...")
        
        # Run face recognition on captured frame
        _, results = self.recognize_face(captured_frame, use_scale=True)
        
        # Display results
        print("\n" + "=" * 50)
        print("  FACE RECOGNITION RESULT")
        print("=" * 50)
        print(f"  Plat: {detected_plate}")
        
        if results and len(results) > 0:
            # Face detected
            detected_name = results[0]['name']
            similarity = results[0]['similarity']
            recognized = results[0]['recognized']
            
            if recognized:
                print(f"  ✅ Wajah terdeteksi: {detected_name}")
                print(f"  📊 Similarity: {similarity:.2f}")
            else:
                print(f"  ⚠️ Wajah tidak dikenali (Unknown)")
                print(f"  📊 Best similarity: {similarity:.2f} (below threshold)")
                detected_name = "Unknown"
        else:
            # No face detected
            print(f"  ❌ Tidak ada wajah terdeteksi dalam gambar")
            detected_name = None
        
        print("=" * 50)
        
        # ===== VERIFICATION LOGIC =====
        if TESTING_MODE:
            # TESTING MODE: Always grant
            print("\n⚠️ [TESTING MODE] Auto-granting access for testing purposes...")
            print("   To enable real verification, set TESTING_MODE = False")
            
            log_name = detected_name if detected_name else "NO_FACE_DETECTED"
            db.log_access(log_name, detected_plate, "GRANTED_TEST_MODE")
            
            print("\n✅ ACCESS GRANTED (Testing Mode)")
            return "GRANTED"
        
        # PRODUCTION MODE: Real verification
        if not detected_name:
            # No face detected
            print("\n❌ ACCESS DENIED - No face detected")
            db.log_access("NO_FACE", detected_plate, "DENIED_NO_FACE")
            return "NO_FACE"
        
        if detected_name == "Unknown":
            # Face detected but not recognized
            print("\n❌ ACCESS DENIED - Face not recognized")
            db.log_access("UNKNOWN", detected_plate, "DENIED_UNKNOWN")
            return "DENIED"
        
        # Face recognized - check against plate owner
        if expected_owner:
            # Normalize names untuk comparison
            detected_lower = detected_name.lower()
            expected_lower = [name.lower() for name in expected_owner] if isinstance(expected_owner, list) else [expected_owner.lower()]
            
            if detected_lower in expected_lower:
                # MATCH: Wajah sesuai dengan pemilik plat
                print(f"\n✅ ACCESS GRANTED - Face matches plate owner")
                print(f"   Verified: {detected_name}")
                db.log_access(detected_name, detected_plate, "GRANTED")
                return "GRANTED"
            else:
                # MISMATCH: Wajah tidak sesuai dengan pemilik plat
                print(f"\n❌ ACCESS DENIED - Face mismatch")
                print(f"   Detected: {detected_name}")
                print(f"   Expected: {', '.join(expected_owner)}")
                db.log_access(detected_name, detected_plate, "DENIED_MISMATCH")
                return "DENIED"
        else:
            # Plat tidak terdaftar, tapi wajah dikenali (surveillance mode)
            print(f"\n⚠️ UNREGISTERED PLATE - Face recognized: {detected_name}")
            print(f"   Allowing access (surveillance mode)")
            db.log_access(detected_name, detected_plate, "GRANTED_UNREGISTERED")
            return "GRANTED"
    
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

