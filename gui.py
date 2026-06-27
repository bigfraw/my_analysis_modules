import PyQt5 as qt
import serial

# function for getting data from a specified serial port
def get_serial_data(port, baudrate=9600, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        data = ser.read(100)  # read up to 100 bytes
        ser.close()
        return data
    except serial.SerialException as e:
        print(f"Error accessing serial port {port}: {e}")
        return None
    
# function to create a simple PyQt5 window
def create_window():
    app = qt.QtWidgets.QApplication([])
    window = qt.QtWidgets.QWidget()
    window.setWindowTitle('Simple PyQt5 Window')
    window.setGeometry(100, 100, 300, 200)
    label = qt.QtWidgets.QLabel('Hello, PyQt5!', parent=window)
    label.move(100, 80)
    window.show()
    app.exec_()

if __name__ == "__main__":
    create_window()
    
