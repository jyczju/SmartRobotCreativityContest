from PySide2 import QtWidgets
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QImage
import cv2
import sys

main_win = None

class NormalWin:

    def __init__(self):
        super().__init__()
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\normal.ui')
        self.ui = QUiLoader().load('normal.ui')
        self.ui.back.clicked.connect(self.main)

    def main(self):
        global main_win
        # 实例化另外一个窗口
        main_win = WindowMain()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


class DefenseWin:

    def __init__(self):
        super().__init__()
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\defense.ui')
        self.ui = QUiLoader().load('defense.ui')
        self.ui.back.clicked.connect(self.main)

    def main(self):
        global main_win
        # 实例化另外一个窗口
        main_win = WindowMain()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


class AttackWin:

    def __init__(self):
        super().__init__()
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\attack.ui')
        self.ui = QUiLoader().load('attack.ui')
        self.ui.back.clicked.connect(self.main)

    def main(self):
        global main_win
        # 实例化另外一个窗口
        main_win = WindowMain()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


class BattleWin:

    def __init__(self):
        super().__init__()
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\battle.ui')
        self.ui = QUiLoader().load('battle.ui')
        self.ui.exit.clicked.connect(self.main)

    def main(self):
        global main_win
        # 实例化另外一个窗口
        main_win = WindowMain()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()

    def cr(self):
        global main_win
        # 实例化另外一个窗口
        main_win = CrWin()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


class CrWin:

    def __init__(self):
        super().__init__()
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\cr.ui')
        self.ui = QUiLoader().load('cr.ui')
        self.ui.main.clicked.connect(self.main)

    def main(self):
        global main_win
        # 实例化另外一个窗口
        main_win = WindowMain()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


class WindowMain:

    def __init__(self):
        super().__init__()
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        # self.ui = QUiLoader().load(r'C:\Users\86157\Desktop\University\Miscellaneous\SRCC\UI\main.ui')
        self.ui = QUiLoader().load('main.ui')

        self.ui.normal.clicked.connect(self.normal)
        self.ui.defense.clicked.connect(self.defense)
        self.ui.attack.clicked.connect(self.attack)
        self.ui.start.clicked.connect(self.battle)

    def normal(self):
        global main_win
        # 实例化另外一个窗口
        main_win = NormalWin()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()
    
    def defense(self):
        global main_win
        # 实例化另外一个窗口
        main_win = DefenseWin()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()
    
    def attack(self):
        global main_win
        # 实例化另外一个窗口
        main_win = AttackWin()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()

    def battle(self):
        global main_win
        # 实例化另外一个窗口
        main_win = BattleWin()
        # 显示新窗口
        main_win.ui.show()
        # 关闭自己
        self.ui.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wm = WindowMain()
    wm.ui.show()
    sys.exit(app.exec_())
