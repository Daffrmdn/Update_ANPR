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
SERVO_PIN = 18      # PWM untuk servo (gate palang) - LANGSUNG KE SERVO
RELAY1_PIN = 6      # Relay 1 - Standby ON (NO)
RELAY2_PIN = 5      # Relay 2 - Standby ON (NO), buzzer di NC
IR_SENSOR_PIN = 26  # IR sensor untuk deteksi kendaraan lewat
LOCAL_BUTTON_PIN = 23  # Local button untuk manual override

# ==================== KONFIGURASI LCD I2C ====================
LCD_I2C_ADDRESS = 0x27  # Alamat I2C LCD (bisa 0x27 atau 0x3F)
LCD_COLS = 16
LCD_ROWS = 2

# ==================== KONFIGURASI SERVO ====================
SERVO_CLOSED_ANGLE = 145    # Posisi tertutup (0 derajat)
SERVO_OPEN_ANGLE = 55     # Posisi terbuka (90 derajat)
SERVO_FREQUENCY = 50      # 50Hz untuk servo standar

# ==================== KONFIGURASI BUTTON ====================
BUTTON_DEBOUNCE_TIME = 300  # ms - debounce untuk mencegah multiple trigger

class RaspberryPiController:
    def __init__(self):
        """Inisialisasi hardware Raspberry Pi"""
        self.is_initialized = False
        self.gate_is_open = False
        self.monitoring_ir = False
        # self.monitoring_button = False
        self.gate_lock = Lock()
        
        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup Servo (PWM) - Langsung tanpa relay
            GPIO.setup(SERVO_PIN, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQUENCY)
            self.servo_pwm.start(0)
            
            # Setup Relay - STANDBY ON (NO aktif, buzzer silent)
            GPIO.setup(RELAY1_PIN, GPIO.OUT)
            GPIO.setup(RELAY2_PIN, GPIO.OUT)
            GPIO.output(RELAY1_PIN, GPIO.LOW)  # Relay 1 STANDBY ON
            GPIO.output(RELAY2_PIN, GPIO.LOW)  # Relay 2 STANDBY ON (buzzer silent)
            print("   ðŸ”Œ Relay 1 & 2 STANDBY ON (buzzer silent)")
            
            # Setup IR Sensor (input dengan pull-up)
            GPIO.setup(IR_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Setup Local Button (input dengan pull-up, button ground saat ditekan)
            GPIO.setup(LOCAL_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print("   ðŸ”˜ Local button initialized (GPIO 23)")
            time.sleep(0.5)  # ? TAMBAHKAN INI (stabilisasi pin)

            GPIO.add_event_detect(
                LOCAL_BUTTON_PIN,
                GPIO.FALLING,
                callback=self._local_button_callback,
                bouncetime=300
            )
            
            # Setup LCD I2C
            try:
                self.lcd = CharLCD(
                    i2c_expander='PCF8574',
                    address=LCD_I2C_ADDRESS,
                    port=1,  # I2C port (biasanya 1 di Raspberry Pi)
                    cols=LCD_COLS,
                    rows=LCD_ROWS,
                    dotsize=8
                )
                self.lcd.clear()
                self.lcd_available = True
                print("   âœ… LCD I2C initialized")
            except Exception as e:
                print(f"   âš ï¸ LCD I2C not available: {e}")
                self.lcd_available = False
            
            self.is_initialized = True
            print("   âœ… Raspberry Pi Controller initialized")
            
            # Set posisi awal gate (tertutup)
            self.close_gate()
            self.display_status("SYSTEM READY", "Waiting...")
            
            # Start monitoring local button
            # self._start_button_monitoring()
            
        except Exception as e:
            print(f"âŒ Error initializing Raspberry Pi Controller: {e}")
            self.is_initialized = False
    
    def _local_button_callback(self, channel):
        time.sleep(0.05)  # debounce manual kecil

         # pastikan masih LOW setelah delay
        if GPIO.input(LOCAL_BUTTON_PIN) != GPIO.LOW:
            return

        print("\n?? LOCAL BUTTON PRESSED!")

        if self.gate_is_open:
            print("   ? Closing gate via local button...")
            self.close_gate()
        else:
            print("   ? Opening gate via local button...")
            self.open_gate(source="BUTTON")
    def _set_servo_angle(self, angle):
        """Set sudut servo (0-180 derajat) - LANGSUNG KIRIM PWM"""
        # Konversi sudut ke duty cycle
        # Duty cycle = (angle / 18) + 2
        # Untuk 0Â° = 2%, untuk 90Â° = 7%, untuk 180Â° = 12%
        duty_cycle = 2.5 + (angle / 18.0) 
        
        # LANGSUNG kirim sinyal PWM ke servo (tanpa relay)
        self.servo_pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Beri waktu servo bergerak
        self.servo_pwm.ChangeDutyCycle(0)  # Stop signal (mencegah jitter)
    
    def open_gate(self, source="SYSTEM"):
        """Membuka gate palang (servo ke 90 derajat)"""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if self.gate_is_open:
                    return True

                print(f"\nOpening gate... (Source: {source})")

                self._set_servo_angle(SERVO_OPEN_ANGLE)

                self.gate_is_open = True
                self.display_status("GATE OPEN", f"Via {source}")
                print("Gate opened!")

                self._start_ir_monitoring()

            return True

        except Exception as e:
            print(f"Error opening gate: {e}")
            return False
        
        
    
    def close_gate(self):
        """Menutup gate palang (servo ke 0 derajat)"""
        if not self.is_initialized:
            return False

        try:
            with self.gate_lock:
                if not self.gate_is_open:
                    return True

                print("\nClosing gate...")

                self._set_servo_angle(SERVO_CLOSED_ANGLE)

                self.gate_is_open = False
                self.display_status("GATE CLOSED", "Waiting...")
                print("Gate closed!")

                self.monitoring_ir = False

            return True

        except Exception as e:
            print(f"Error closing gate: {e}")
            return False
    
    def _start_ir_monitoring(self):
        """Mulai monitoring IR sensor di background thread"""
        if self.monitoring_ir:
            return  # Sudah monitoring
        
        self.monitoring_ir = True
        
        def monitor_ir():
            print("ðŸ‘ï¸ IR Sensor monitoring started...")
            vehicle_detected = False
            
            while self.monitoring_ir and self.gate_is_open:
                # Baca status IR sensor
                # IR sensor biasanya LOW ketika ada objek terdeteksi
                ir_state = GPIO.input(IR_SENSOR_PIN)
                
                if ir_state == GPIO.LOW and not vehicle_detected:
                    # Kendaraan terdeteksi melewati gate
                    print("ðŸš— Vehicle detected passing through gate!")
                    self.display_status("VEHICLE PASSING", "Please Wait...")
                    vehicle_detected = True
                
                elif ir_state == GPIO.HIGH and vehicle_detected:
                    # Kendaraan sudah lewat, tunggu sebentar lalu tutup gate
                    print("âœ… Vehicle passed, closing gate in 2 seconds...")
                    time.sleep(2)
                    self.close_gate()
                    break
                
                time.sleep(0.1)  # Check setiap 100ms
            
            print("ðŸ‘ï¸ IR Sensor monitoring stopped.")
        
        # Jalankan monitoring di thread terpisah
        ir_thread = Thread(target=monitor_ir, daemon=True)
        ir_thread.start()
    
    # def _start_button_monitoring(self):
    #     """Mulai monitoring local button di background thread"""
    #     if self.monitoring_button:
    #         return  # Sudah monitoring
        
    #     self.monitoring_button = True
        
    #     def monitor_button():
    #         print("ðŸ”˜ Local button monitoring started...")
    #         last_press_time = 0
            
    #         while self.monitoring_button:
    #             # Baca status button (LOW = ditekan, HIGH = tidak ditekan)
    #             button_state = GPIO.input(LOCAL_BUTTON_PIN)
    #             current_time = time.time() * 1000  # Convert to milliseconds
                
    #             if button_state == GPIO.LOW:  # Button ditekan
    #                 # Debounce check
    #                 if current_time - last_press_time > BUTTON_DEBOUNCE_TIME:
    #                     last_press_time = current_time
                        
    #                     print("\nðŸ”˜ LOCAL BUTTON PRESSED!")
                        
    #                     if not self.gate_is_open:
    #                         # Gate tertutup, buka gate
    #                         print("   â†’ Opening gate via local button...")
    #                         self.display_status("MANUAL OPEN", "Local Button")
    #                         time.sleep(0.5)  # Delay sebentar untuk user melihat LCD
    #                         self.open_gate(source="BUTTON")
    #                     else:
    #                         # Gate terbuka, tutup gate
    #                         print("   â†’ Closing gate via local button...")
    #                         self.display_status("MANUAL CLOSE", "Local Button")
    #                         time.sleep(0.5)
    #                         self.close_gate()
                        
    #                     # Tunggu button dilepas
    #                     while GPIO.input(LOCAL_BUTTON_PIN) == GPIO.LOW:
    #                         time.sleep(0.05)
                
    #             time.sleep(0.05)  # Check setiap 50ms
            
    #         print("ðŸ”˜ Local button monitoring stopped.")
        
    #     # Jalankan monitoring di thread terpisah
    #     button_thread = Thread(target=monitor_button, daemon=True)
    #     button_thread.start()
    
    def display_status(self, line1, line2=""):
        """Tampilkan status di LCD I2C"""
        if not self.lcd_available:
            return
        
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1[:LCD_COLS])  # Maksimal sesuai kolom LCD
            
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:LCD_COLS])
        except Exception as e:
            print(f"âš ï¸ LCD error: {e}")
    
    def access_granted_sequence(self):
        """Sequence lengkap saat akses diterima"""
        if not self.is_initialized:
            print("âš ï¸ Controller not initialized!")
            return
        
        try:
            # 1. Tampilkan pesan di LCD
            self.display_status("ACCESS GRANTED", "Opening Gate...")
            
            # 2. Buka gate (LANGSUNG kirim PWM ke servo)
            print("ðŸŽ›ï¸ Sending PWM signal to servo...")
            self.open_gate(source="SYSTEM")
            
            # IR sensor akan otomatis menutup gate setelah kendaraan lewat
            
        except Exception as e:
            print(f"âŒ Error in access granted sequence: {e}")
    
    def access_denied_sequence(self):
        """Sequence saat akses ditolak"""
        if not self.is_initialized:
            return
        
        try:
            # Tampilkan pesan di LCD
            self.display_status("ACCESS DENIED", "Go Away!")
            
            time.sleep(2)
            self.display_status("SYSTEM READY", "Waiting...")
            
        except Exception as e:
            print(f"âŒ Error in access denied sequence: {e}")
    
    def cleanup(self):
        """Cleanup GPIO dan LCD saat program selesai"""
        try:
            print("\nðŸ§¹ Cleaning up GPIO...")
            
            # Stop monitoring
            # self.monitoring_button = False
            self.monitoring_ir = False
            
            # Tutup gate jika masih terbuka
            if self.gate_is_open:
                self.close_gate()
            
            # Stop PWM
            if hasattr(self, 'servo_pwm'):
                self.servo_pwm.stop()
            
            # MATIKAN RELAY (buzzer akan bunyi sebagai tanda sistem shutdown)
            print("   âš ï¸ Shutting down relays (buzzer will sound - emergency mode)")
            GPIO.output(RELAY1_PIN, GPIO.HIGH)
            GPIO.output(RELAY2_PIN, GPIO.HIGH)
            
            # Clear LCD
            if self.lcd_available:
                self.lcd.clear()
                self.lcd.write_string("System Shutdown")
            
            time.sleep(1)  # Beri waktu buzzer bunyi sebentar
            
            # Cleanup GPIO
            GPIO.cleanup()
            print("   âœ… Cleanup complete")
            
        except Exception as e:
            print(f"âš ï¸ Error during cleanup: {e}")


# ==================== TEST PROGRAM ====================
if __name__ == "__main__":
    print("=" * 50)
    print("  RASPBERRY PI CONTROLLER TEST")
    print("=" * 50)
    
    controller = RaspberryPiController()
    
    if not controller.is_initialized:
        print("âŒ Controller initialization failed!")
        exit(1)
    
    try:
        print("\nTesting hardware components...")
        
        # Test 1: LCD Display
        print("\n1. Testing LCD...")
        controller.display_status("LCD TEST", "Hello World!")
        time.sleep(2)
        
        # Test 2: Check Relay Status (should be ON)
        print("\n2. Checking Relay Status...")
        relay1_state = GPIO.input(RELAY1_PIN)
        relay2_state = GPIO.input(RELAY2_PIN)
        print(f"   Relay 1: {'ON (HIGH)' if relay1_state else 'OFF (LOW)'}")
        print(f"   Relay 2: {'ON (HIGH)' if relay2_state else 'OFF (LOW)'}")
        print(f"   Buzzer: {'Silent' if relay2_state else 'SOUNDING'}")
        time.sleep(2)
        
        # Test 3: Gate Open/Close
        print("\n3. Testing Gate (Direct PWM)...")
        controller.display_status("GATE TEST", "Opening...")
        controller.open_gate(source="TEST")
        time.sleep(5)
        
        print("\n   Closing gate manually (for testing)...")
        controller.close_gate()
        time.sleep(2)
        
        # Test 4: Local Button
        print("\n4. Testing Local Button...")
        print("   Press the local button (GPIO 23) to toggle gate...")
        print("   Waiting 15 seconds for button press...")
        controller.display_status("BUTTON TEST", "Press Button!")
        time.sleep(15)
        
        if controller.gate_is_open:
            print("   Closing gate after button test...")
            controller.close_gate()
        
        # Test 5: Full sequence
        print("\n5. Testing Access Granted Sequence...")
        controller.access_granted_sequence()
        print("   Waiting for IR sensor or manual close...")
        print("   (Simulate IR sensor by covering it or wait 10 seconds)")
        time.sleep(10)
        
        if controller.gate_is_open:
            print("   Closing gate manually...")
            controller.close_gate()
        
        # Test 6: Access Denied
        print("\n6. Testing Access Denied Sequence...")
        controller.access_denied_sequence()
        time.sleep(2)
        
        # Test 7: Emergency simulation (matikan relay sebentar)
        print("\n7. Testing Emergency Mode (relay OFF = buzzer ON)...")
        print("   WARNING: Buzzer will sound!")
        controller.display_status("EMERGENCY TEST", "Buzzer ON...")
        GPIO.output(RELAY2_PIN, GPIO.HIGH)  # Matikan relay 2 = buzzer bunyi
        time.sleep(2)
        GPIO.output(RELAY2_PIN, GPIO.LOW)  # Nyalakan kembali = buzzer silent
        print("   Relay 2 back to standby ON")
        
        print("\nâœ… All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\nâ¹ Test interrupted by user")
    except Exception as e:
        print(f"\nâŒ Error during test: {e}")
    finally:
        controller.cleanup()
        print("\nðŸ‘‹ Test finished!")
