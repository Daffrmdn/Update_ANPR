"""
Raspberry Pi Hardware Controller
Mengontrol Servo, Relay, Buzzer, IR Sensor, LCD I2C, dan Local Button
Untuk sistem akses kontrol plat nomor + face recognition
"""

import RPi.GPIO as GPIO
import time
import smbus2
from RPLCD.i2c import CharLCD
from threading import Thread, Lock

# ==================== KONFIGURASI PIN ====================
SERVO_PIN = 18          # PWM untuk servo (gate palang) - via NO Relay ke Servo
RELAY1_PIN = 6          # Relay 1 - Standby ON (NO)
RELAY2_PIN = 5          # Relay 2 - Standby ON (NO), buzzer di NC
IR_SENSOR_PIN = 26      # IR sensor untuk deteksi kendaraan lewat
LOCAL_BUTTON_PIN = 23   # Local button untuk manual override

# ==================== KONFIGURASI LCD I2C ====================
LCD_I2C_ADDRESS = 0x27  # Alamat I2C LCD (bisa 0x27 atau 0x3F)
LCD_COLS = 16
LCD_ROWS = 2

# ==================== KONFIGURASI SERVO ====================
SERVO_CLOSED_ANGLE = 145    # Posisi tertutup
SERVO_OPEN_ANGLE = 55       # Posisi terbuka
SERVO_FREQUENCY = 50        # 50Hz untuk servo standar

# ==================== KONFIGURASI BUTTON ====================
BUTTON_DEBOUNCE_TIME = 300  # ms


class RaspberryPiController:
    def __init__(self):
        """Inisialisasi hardware Raspberry Pi"""
        self.is_initialized = False
        self.gate_is_open = False
        self.monitoring_ir = False
        self.gate_lock = Lock()

        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # ---- Servo PWM (Pin 18 → NO Relay → Servo) ----
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQUENCY)
            self.servo_pwm.start(0)

            # ---- Relay (STANDBY ON = LOW) ----
            GPIO.setup(RELAY1_PIN, GPIO.OUT)
            GPIO.setup(RELAY2_PIN, GPIO.OUT)
            GPIO.output(RELAY1_PIN, GPIO.LOW)   # Relay 1 STANDBY ON
            GPIO.output(RELAY2_PIN, GPIO.LOW)   # Relay 2 STANDBY ON (buzzer silent)
            print("   🔌 Relay 1 & 2 STANDBY ON (buzzer silent)")

            # ---- IR Sensor ----
            GPIO.setup(IR_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # ---- Local Button (Pin 23) ----
            GPIO.setup(LOCAL_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print("   🔘 Local button initialized (GPIO 23)")

            # Stabilisasi pin sebelum pasang event detect
            time.sleep(0.5)

            # Hapus event detect lama jika ada (mencegah konflik saat restart)
            try:
                GPIO.remove_event_detect(LOCAL_BUTTON_PIN)
            except Exception:
                pass

            # Gunakan event detect dengan callback yang spawn thread
            # agar tidak terjadi deadlock dengan gate_lock
            GPIO.add_event_detect(
                LOCAL_BUTTON_PIN,
                GPIO.FALLING,
                callback=self._local_button_callback,
                bouncetime=BUTTON_DEBOUNCE_TIME
            )
            print("   🔘 Local button event detect registered")

            # ---- LCD I2C ----
            try:
                self.lcd = CharLCD(
                    i2c_expander='PCF8574',
                    address=LCD_I2C_ADDRESS,
                    port=1,
                    cols=LCD_COLS,
                    rows=LCD_ROWS,
                    dotsize=8
                )
                self.lcd.clear()
                self.lcd_available = True
                print("   ✅ LCD I2C initialized")
            except Exception as e:
                print(f"   ⚠️ LCD I2C not available: {e}")
                self.lcd_available = False

            self.is_initialized = True
            print("   ✅ Raspberry Pi Controller initialized")

            # Set posisi awal gate (tertutup)
            self._move_servo_to_closed()
            self.display_status("SYSTEM READY", "Waiting...")

        except Exception as e:
            print(f"❌ Error initializing Raspberry Pi Controller: {e}")
            self.is_initialized = False

    # ------------------------------------------------------------------
    # PRIVATE: Servo movement (tanpa lock — hanya dipanggil dari dalam lock)
    # ------------------------------------------------------------------

    def _set_servo_angle(self, angle):
        """Set sudut servo (0-180°). Kirim PWM lalu stop untuk cegah jitter."""
        duty_cycle = 2.5 + (angle / 18.0)
        self.servo_pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)                       # beri waktu servo bergerak
        self.servo_pwm.ChangeDutyCycle(0)     # hentikan sinyal (anti-jitter)

    def _move_servo_to_closed(self):
        """Helper: gerak ke posisi tutup TANPA lock (untuk init)."""
        self._set_servo_angle(SERVO_CLOSED_ANGLE)
        self.gate_is_open = False

    # ------------------------------------------------------------------
    # PRIVATE: Local button callback
    # Callback GPIO interrupt TIDAK boleh blocking / memakai lock langsung.
    # Solusi: spawn daemon thread yang memanggil open_gate / close_gate.
    # ------------------------------------------------------------------

    def _local_button_callback(self, channel):
        """Dipanggil oleh GPIO event detect saat tombol ditekan.
        LANGSUNG gerak servo tanpa menunggu gate_lock agar tidak deadlock
        dengan main loop yang sedang memegang lock."""
        # Debounce manual kecil
        time.sleep(0.05)

        # Pastikan pin masih LOW (bukan noise)
        if GPIO.input(LOCAL_BUTTON_PIN) != GPIO.LOW:
            return

        print("\n🔘 LOCAL BUTTON PRESSED!")

        def _handle_button():
            if self.gate_is_open:
                print("   ↓ Closing gate via local button (direct)...")
                self._set_servo_angle(SERVO_CLOSED_ANGLE)
                self.gate_is_open = False
                self.monitoring_ir = False
                self.display_status("GATE CLOSED", "Via BUTTON")
                print("   ✅ Gate closed via button")
            else:
                print("   ↑ Opening gate via local button (direct)...")
                self._set_servo_angle(SERVO_OPEN_ANGLE)
                self.gate_is_open = True
                self.display_status("GATE OPEN", "Via BUTTON")
                print("   ✅ Gate opened via button")
                # Start IR monitoring setelah buka
                self._start_ir_monitoring()

        t = Thread(target=_handle_button, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # PUBLIC: Gate control
    # ------------------------------------------------------------------

    def open_gate(self, source="SYSTEM"):
        """Membuka gate palang."""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if self.gate_is_open:
                    print("   ℹ️ Gate already open.")
                    return True

                print(f"\n🔓 Opening gate... (Source: {source})")
                self._set_servo_angle(SERVO_OPEN_ANGLE)
                self.gate_is_open = True

                # LCD: tampilkan status GATE OPEN
                self.display_status("GATE OPEN", f"Via {source[:14]}")
                print("✅ Gate opened!")

            # Mulai IR monitoring di luar lock
            self._start_ir_monitoring()
            return True

        except Exception as e:
            print(f"❌ Error opening gate: {e}")
            return False

    def close_gate(self):
        """Menutup gate palang."""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if not self.gate_is_open:
                    print("   ℹ️ Gate already closed.")
                    return True

                print("\n🔒 Closing gate...")

                # Hentikan IR monitoring sebelum gerak servo
                self.monitoring_ir = False

                self._set_servo_angle(SERVO_CLOSED_ANGLE)
                self.gate_is_open = False

                # LCD: tampilkan status GATE CLOSED
                self.display_status("GATE CLOSED", "Waiting...")
                print("✅ Gate closed!")

            return True

        except Exception as e:
            print(f"❌ Error closing gate: {e}")
            return False

    # ------------------------------------------------------------------
    # PRIVATE: IR Sensor monitoring
    # ------------------------------------------------------------------

    def _start_ir_monitoring(self):
        """Mulai monitoring IR sensor di background thread."""
        if self.monitoring_ir:
            return

        self.monitoring_ir = True

        def monitor_ir():
            print("👁️ IR Sensor monitoring started...")
            vehicle_detected = False

            while self.monitoring_ir and self.gate_is_open:
                ir_state = GPIO.input(IR_SENSOR_PIN)

                if ir_state == GPIO.LOW and not vehicle_detected:
                    # Kendaraan terdeteksi memasuki sensor
                    print("🚗 Vehicle detected passing through gate!")
                    self.display_status("VEHICLE PASSING", "Please Wait...")
                    vehicle_detected = True

                elif ir_state == GPIO.HIGH and vehicle_detected:
                    # Kendaraan sudah melewati sensor
                    print("✅ Vehicle passed! Closing gate in 2 seconds...")
                    self.display_status("VEHICLE PASSED", "Closing...")
                    time.sleep(2)
                    self.close_gate()
                    break

                time.sleep(0.1)

            print("👁️ IR Sensor monitoring stopped.")

        ir_thread = Thread(target=monitor_ir, daemon=True)
        ir_thread.start()

    # ------------------------------------------------------------------
    # PUBLIC: LCD display
    # ------------------------------------------------------------------

    def display_status(self, line1, line2=""):
        """Tampilkan status di LCD I2C (max LCD_COLS karakter per baris)."""
        if not self.lcd_available:
            # Fallback ke console jika LCD tidak tersedia
            print(f"   [LCD] {line1} | {line2}")
            return

        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1[:LCD_COLS])

            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:LCD_COLS])
        except Exception as e:
            print(f"⚠️ LCD error: {e}")

    # ------------------------------------------------------------------
    # PUBLIC: Sequence helpers (dipanggil dari main.py)
    # ------------------------------------------------------------------

    def access_granted_sequence(self):
        """Sequence lengkap saat akses diterima."""
        if not self.is_initialized:
            print("⚠️ Controller not initialized!")
            return

        try:
            self.display_status("ACCESS GRANTED", "Opening Gate...")
            print("🎛️ Sending PWM signal to servo...")
            self.open_gate(source="SYSTEM")
            # IR sensor akan otomatis menutup gate setelah kendaraan lewat

        except Exception as e:
            print(f"❌ Error in access granted sequence: {e}")

    def access_denied_sequence(self):
        """Sequence saat akses ditolak."""
        if not self.is_initialized:
            return

        try:
            self.display_status("ACCESS DENIED", "Go Away!")
            time.sleep(2)
            self.display_status("SYSTEM READY", "Waiting...")

        except Exception as e:
            print(f"❌ Error in access denied sequence: {e}")

    # ------------------------------------------------------------------
    # PUBLIC: Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Cleanup GPIO dan LCD saat program selesai."""
        try:
            print("\n🧹 Cleaning up GPIO...")

            # Hentikan monitoring
            self.monitoring_ir = False

            # Tutup gate jika masih terbuka
            if self.gate_is_open:
                self.close_gate()

            # Stop PWM
            if hasattr(self, 'servo_pwm'):
                self.servo_pwm.stop()

            # Matikan relay (buzzer bunyi sebagai tanda shutdown)
            print("   ⚠️ Shutting down relays (buzzer will sound - shutdown signal)")
            GPIO.output(RELAY1_PIN, GPIO.HIGH)
            GPIO.output(RELAY2_PIN, GPIO.HIGH)

            # Clear LCD
            if self.lcd_available:
                self.lcd.clear()
                self.lcd.write_string("System Shutdown")

            time.sleep(1)

            GPIO.cleanup()
            print("   ✅ Cleanup complete")

        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


# ==================== TEST PROGRAM ====================
if __name__ == "__main__":
    print("=" * 50)
    print("  RASPBERRY PI CONTROLLER TEST")
    print("=" * 50)

    controller = RaspberryPiController()

    if not controller.is_initialized:
        print("❌ Controller initialization failed!")
        exit(1)

    try:
        print("\nTesting hardware components...")

        # Test 1: LCD Display
        print("\n1. Testing LCD...")
        controller.display_status("LCD TEST", "Hello World!")
        time.sleep(2)

        # Test 2: Relay Status
        print("\n2. Checking Relay Status...")
        relay1_state = GPIO.input(RELAY1_PIN)
        relay2_state = GPIO.input(RELAY2_PIN)
        print(f"   Relay 1: {'ON (HIGH)' if relay1_state else 'STANDBY (LOW)'}")
        print(f"   Relay 2: {'ON (HIGH)' if relay2_state else 'STANDBY (LOW)'}")
        print(f"   Buzzer: {'Silent' if not relay2_state else 'SOUNDING'}")
        time.sleep(2)

        # Test 3: Gate Open/Close
        print("\n3. Testing Gate (Direct PWM via Relay)...")
        controller.display_status("GATE TEST", "Opening...")
        controller.open_gate(source="TEST")
        time.sleep(5)

        print("\n   Closing gate manually (for testing)...")
        controller.close_gate()
        time.sleep(2)

        # Test 4: Local Button
        print("\n4. Testing Local Button (GPIO 23)...")
        print("   Press the local button to toggle gate (waiting 15 seconds)...")
        controller.display_status("BUTTON TEST", "Press Button!")
        time.sleep(15)

        if controller.gate_is_open:
            print("   Closing gate after button test...")
            controller.close_gate()

        # Test 5: Full Access Granted Sequence
        print("\n5. Testing Access Granted Sequence...")
        controller.access_granted_sequence()
        print("   Waiting for IR sensor or 10 seconds timeout...")
        time.sleep(10)

        if controller.gate_is_open:
            print("   Closing gate manually...")
            controller.close_gate()

        # Test 6: Access Denied
        print("\n6. Testing Access Denied Sequence...")
        controller.access_denied_sequence()
        time.sleep(2)

        # Test 7: Emergency buzzer simulation
        print("\n7. Testing Emergency Mode (relay OFF = buzzer ON)...")
        print("   WARNING: Buzzer will sound for 2 seconds!")
        controller.display_status("EMERGENCY TEST", "Buzzer ON...")
        GPIO.output(RELAY2_PIN, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(RELAY2_PIN, GPIO.LOW)
        controller.display_status("SYSTEM READY", "Waiting...")
        print("   Relay 2 back to standby ON")

        print("\n✅ All tests completed!")

    except KeyboardInterrupt:
        print("\n\n⏹ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        controller.cleanup()
        print("\n👋 Test finished!")
