#!/usr/bin/env python3
import serial
import time
import sys

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/tty.usbmodem1401" 
BAUD_RATE = 9600

def clean_exit(arduino_instance, message):
    """Sends lowercase commands to turn off everything before closing the port."""
    print(message)
    try:
        print("🧹 Cleaning up: Turning off all LEDs and Buzzer...")
        # Send lowercase codes to turn off Green, Red/Buzzer, and Blue
        arduino_instance.write(b'g')
        arduino_instance.write(b'r')
        arduino_instance.write(b'b')
        time.sleep(0.1) # Give Arduino a tiny moment to process bytes
    except Exception as e:
        print(f"Could not send cleanup commands: {e}")
    finally:
        arduino_instance.close()
        print("🔌 Serial port closed securely. Bye!")

def run_test():
    print("=" * 60)
    print("   ARDUINO RADAR ALARM - CASE-SENSITIVE PROTOCOL TESTER   ")
    print("=" * 60)
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    
    try:
        arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
        time.sleep(2) 
        print("✅ Connected successfully!\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    print("--- AVAILABLE COMMANDS ---")
    print("  [G] -> Turn ON Green LED  |  [g] -> Turn OFF Green LED")
    print("  [R] -> Turn ON Red + Buzz |  [r] -> Turn OFF Red + Buzz")
    print("  [B] -> Turn ON Blue LED   |  [b] -> Turn OFF Blue LED")
    print("  [Q] -> Quit the tester")
    print("-" * 60)

    while True:
        try:
            user_input = input("Enter command: ").strip()
            
            # Handle clean exit via command
            if user_input.lower() == 'q':
                clean_exit(arduino, "\nExiting tester via user request.")
                break
                
            if user_input in ['G', 'g', 'R', 'r', 'B', 'b']:
                command_byte = user_input.encode('utf-8')
                arduino.write(command_byte)
                print(f"➡️ Sent to Arduino: '{user_input}'")
            else:
                print("⚠️ Invalid character! Use G/g, R/r, B/b, or Q to quit.")
                
        except KeyboardInterrupt:
            # Handle clean exit via Ctrl+C
            clean_exit(arduino, "\n\n🚨 KeyboardInterrupt detected!")
            break
        except Exception as e:
            print(f"❌ Error during communication: {e}")
            break

if __name__ == "__main__":
    run_test()