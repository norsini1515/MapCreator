from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtWidgets import QWidget, QApplication
'''
launched qt designer with: pyside6-designer
created a simple ui with a button and a label, saved as test.ui

'''
if __name__ == "__main__":
    app = QApplication([])
    loader = QUiLoader()
    ui_file = QFile("test.ui")
    ui_file.open(QFile.ReadOnly)
    window = loader.load(ui_file)
    ui_file.close()

    window.show()
    app.exec()