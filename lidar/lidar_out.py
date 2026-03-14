import serial

ser = serial.Serial('/dev/serial0', 115200, timeout=1)

while True:
    if ser.read() == b'\x59':
        if ser.read() == b'\x59':
            low = ord(ser.read())
            high = ord(ser.read())
            distance = low + high * 256
            ser.read(5)
            print("Distance:", distance, "cm")