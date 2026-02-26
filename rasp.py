"""
Raspberry Pi Hardware Controller
Mengontrol Servo, Relay, Buzzer, IR Sensor, LCD I2C, dan Local Button
Untuk sistem akses kontrol plat nomor + face recognition

CATATAN: Menggunakan polling thread untuk button (bukan GPIO.add_event_detect)
karena event detect tidak reliable di Python 3.13 + RPi.GPIO versi lama
"""

import RPi.GPIO as GPIO
import time
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
BUTTON_DEBOUNCE_MS = 300    # ms debounce


class RaspberryPiController:
    def __init__(self):
        """Inisialisasi hardware Raspberry Pi"""
        self.is_initialized = False
        self.gate_is_open = False
        self.monitoring_ir = False
        self._button_polling_active = False
        self.gate_lock = Lock()

        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # ---- Servo PWM (Pin 18 -> NO Relay -> Servo) ----
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQUENCY)
            self.servo_pwm.start(0)

            # ---- Relay (STANDBY ON = LOW) ----
            GPIO.setup(RELAY1_PIN, GPIO.OUT)
            GPIO.setup(RELAY2_PIN, GPIO.OUT)
            GPIO.output(RELAY1_PIN, GPIO.LOW)
            GPIO.output(RELAY2_PIN, GPIO.LOW)
            print("   Relay 1 & 2 STANDBY ON (buzzer silent)")

            # ---- IR Sensor ----
            GPIO.setup(IR_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # ---- Local Button (Pin 23) ----
            GPIO.setup(LOCAL_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print("   Local button initialized (GPIO 23)")

            # Stabilisasi pin
            time.sleep(0.5)

            # Start polling thread untuk button
            # Lebih reliable daripada GPIO.add_event_detect di Python 3.13
            self._button_polling_active = True
            self._button_thread = Thread(target=self._poll_button, daemon=True)
            self._button_thread.start()
            print("   Local button polling thread started")

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
                print("   LCD I2C initialized")
            except Exception as e:
                print(f"   LCD I2C not available: {e}")
                self.lcd_available = False

            self.is_initialized = True
            print("   Raspberry Pi Controller initialized")

            # Set posisi awal gate (tertutup)
            self._move_servo_to_closed()
            self.display_status("SYSTEM READY", "Waiting...")

        except Exception as e:
            print(f"Error initializing Raspberry Pi Controller: {e}")
            self.is_initialized = False

    # ------------------------------------------------------------------
    # PRIVATE: Servo
    # ------------------------------------------------------------------

    def _set_servo_angle(self, angle):
        """Set sudut servo. Kirim PWM lalu stop untuk cegah jitter."""
        duty_cycle = 2.5 + (angle / 18.0)
        self.servo_pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.8)
        self.servo_pwm.ChangeDutyCycle(0)

    def _move_servo_to_closed(self):
        """Gerak ke posisi tutup tanpa lock (untuk init)."""
        self._set_servo_angle(SERVO_CLOSED_ANGLE)
        self.gate_is_open = False

    # ------------------------------------------------------------------
    # PRIVATE: Button polling thread
    # Poll setiap 50ms, deteksi FALLING edge (HIGH->LOW)
    # Lebih reliable daripada GPIO.add_event_detect di Python 3.13
    # ------------------------------------------------------------------

    def _poll_button(self):
        """Thread polling button pin 23 setiap 50ms."""
        print("   Button polling active (pin 23)...")
        last_state = GPIO.HIGH
        last_press_time = 0

        while self._button_polling_active:
            try:
                current_state = GPIO.input(LOCAL_BUTTON_PIN)
                current_time = time.time() * 1000  # ms

                # Deteksi FALLING edge: HIGH -> LOW = button ditekan
                if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                    if current_time - last_press_time > BUTTON_DEBOUNCE_MS:
                        last_press_time = current_time
                        print(f"\nLOCAL BUTTON PRESSED! (polled)")
                        # Handle di thread baru agar polling tidak terhambat
                        Thread(target=self._handle_button_press, daemon=True).start()

                last_state = current_state
                time.sleep(0.05)  # poll setiap 50ms

            except Exception as e:
                print(f"Button polling error: {e}")
                time.sleep(0.1)

        print("   Button polling stopped.")

    def _handle_button_press(self):
        """Handle aksi saat button ditekan.
        Langsung gerak servo TANPA menunggu gate_lock
        agar tidak terhambat main loop."""
        if self.gate_is_open:
            print("   Closing gate via local button...")
            # Stop IR monitoring
            self.monitoring_ir = False
            # Langsung gerak servo tanpa lock
            self._set_servo_angle(SERVO_CLOSED_ANGLE)
            self.gate_is_open = False
            self.display_status("GATE CLOSED", "Via BUTTON")
            print("   Gate closed via button")
        else:
            print("   Opening gate via local button...")
            # Langsung gerak servo tanpa lock
            self._set_servo_angle(SERVO_OPEN_ANGLE)
            self.gate_is_open = True
            self.display_status("GATE OPEN", "Via BUTTON")
            print("   Gate opened via button")
            # Start IR monitoring
            self._start_ir_monitoring()

    # ------------------------------------------------------------------
    # PUBLIC: Gate control (dipanggil dari main.py)
    # ------------------------------------------------------------------

    def open_gate(self, source="SYSTEM"):
        """Membuka gate palang."""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if self.gate_is_open:
                    print("   Gate already open.")
                    return True

                print(f"\nOpening gate... (Source: {source})")
                self._set_servo_angle(SERVO_OPEN_ANGLE)
                self.gate_is_open = True
                self.display_status("GATE OPEN", f"Via {source[:14]}")
                print("Gate opened!")

            self._start_ir_monitoring()
            return True

        except Exception as e:
            print(f"Error opening gate: {e}")
            return False

    def close_gate(self):
        """Menutup gate palang."""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if not self.gate_is_open:
                    print("   Gate already closed.")
                    return True

                print("\nClosing gate...")
                self.monitoring_ir = False
                self._set_servo_angle(SERVO_CLOSED_ANGLE)
                self.gate_is_open = False
                self.display_status("GATE CLOSED", "Waiting...")
                print("Gate closed!")

            return True

        except Exception as e:
            print(f"Error closing gate: {e}")
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
            print("IR Sensor monitoring started...")
            vehicle_detected = False

            while self.monitoring_ir and self.gate_is_open:
                ir_state = GPIO.input(IR_SENSOR_PIN)

                if ir_state == GPIO.LOW and not vehicle_detected:
                    print("Vehicle detected passing through gate!")
                    self.display_status("VEHICLE PASSING", "Please Wait...")
                    vehicle_detected = True

                elif ir_state == GPIO.HIGH and vehicle_detected:
                    print("Vehicle passed! Closing gate in 2 seconds...")
                    self.display_status("VEHICLE PASSED", "Closing...")
                    time.sleep(2)
                    self.close_gate()
                    break

                time.sleep(0.1)

            print("IR Sensor monitoring stopped.")

        ir_thread = Thread(target=monitor_ir, daemon=True)
        ir_thread.start()

    # ------------------------------------------------------------------
    # PUBLIC: LCD display
    # ------------------------------------------------------------------

    def display_status(self, line1, line2=""):
        """Tampilkan status di LCD I2C."""
        if not self.lcd_available:
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
            print(f"LCD error: {e}")

    # ------------------------------------------------------------------
    # PUBLIC: Sequence helpers (dipanggil dari main.py)
    # ------------------------------------------------------------------

    def access_granted_sequence(self):
        """Sequence lengkap saat akses diterima."""
        if not self.is_initialized:
            print("Controller not initialized!")
            return
        try:
            self.display_status("ACCESS GRANTED", "Opening Gate...")
            self.open_gate(source="SYSTEM")
        except Exception as e:
            print(f"Error in access granted sequence: {e}")

    def access_denied_sequence(self):
        """Sequence saat akses ditolak."""
        if not self.is_initialized:
            return
        try:
            self.display_status("ACCESS DENIED", "Go Away!")
            time.sleep(2)
            self.display_status("SYSTEM READY", "Waiting...")
        except Exception as e:
            print(f"Error in access denied sequence: {e}")

    # ------------------------------------------------------------------
    # PUBLIC: Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Cleanup GPIO dan LCD saat program selesai."""
        try:
            print("\nCleaning up GPIO...")

            # Hentikan polling thread dan IR monitoring
            self._button_polling_active = False
            self.monitoring_ir = False

            # Tutup gate jika masih terbuka
            if self.gate_is_open:
                self._set_servo_angle(SERVO_CLOSED_ANGLE)
                self.gate_is_open = False

            # Stop PWM
            if hasattr(self, 'servo_pwm'):
                self.servo_pwm.stop()

            # Matikan relay (buzzer bunyi sebagai tanda shutdown)
            print("   Shutting down relays (buzzer will sound - shutdown signal)")
            GPIO.output(RELAY1_PIN, GPIO.HIGH)
            GPIO.output(RELAY2_PIN, GPIO.HIGH)

            # Clear LCD
            if self.lcd_available:
                self.lcd.clear()
                self.lcd.write_string("System Shutdown")

            time.sleep(1)
            GPIO.cleanup()
            print("   Cleanup complete")

        except Exception as e:
            print(f"Error during cleanup: {e}")


# ==================== TEST PROGRAM ====================
if __name__ == "__main__":
    print("=" * 50)
    print("  RASPBERRY PI CONTROLLER TEST")
    print("=" * 50)

    controller = RaspberryPiController()

    if not controller.is_initialized:
        print("Controller initialization failed!")
        exit(1)

    try:
        print("\nTesting hardware components...")

        # Test 1: LCD
        print("\n1. Testing LCD...")
        controller.display_status("LCD TEST", "Hello World!")
        time.sleep(2)

        # Test 2: Gate Open/Close langsung
        print("\n2. Testing Gate...")
        controller.display_status("GATE TEST", "Opening...")
        controller.open_gate(source="TEST")
        time.sleep(5)
        controller.close_gate()
        time.sleep(2)

        # Test 3: Local Button polling - 20 detik
        print("\n3. Testing Local Button (GPIO 23) - 20 detik...")
        print("   Tekan tombol untuk toggle gate...")
        controller.display_status("BUTTON TEST", "Press Button!")
        time.sleep(20)

        if controller.gate_is_open:
            controller.close_gate()

        print("\nAll tests completed!")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        controller.cleanup()
        print("\nTest finished!")
