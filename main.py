# ML APP/main.py
import sys
from PySide6.QtWidgets import QApplication
from ML_TAB.windows.ML_tab import ML_Tab

def main():
    app = QApplication(sys.argv)
    win = ML_Tab()  # MainWindow sẽ tự nạp base.qss + theme.qss
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
