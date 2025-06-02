"""
Mall Blueprint Mapping Application

Main entry point for the application.
This file imports and runs the main window.
"""

import sys

from gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())