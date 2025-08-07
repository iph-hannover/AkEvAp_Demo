from gui import AkevapVisual
import argparse
import time
import sys
from pathlib import Path
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

parser = argparse.ArgumentParser(description='')
parser.add_argument('--folder', type=str, required=True, help='Trial path')
args = parser.parse_args()

# Single video testing:
# args.folder = ...
# cfg.NO_GUI = False
# cfg.RECORD_GUI = True
# cfg.LOOP_VIDEO = True

# Batch evaluation: (no gui)
# cfg.NO_GUI = True
# cfg.RECORD_GUI = False
# cfg.LOOP_VIDEO = False

# Batch evaluation: (with gui)
# cfg.NO_GUI = False
# cfg.RECORD_GUI = True
# cfg.LOOP_VIDEO = False

import gui as cfg
cfg.NO_GUI = False  # useful for batch evaluation of pipeline on multiple videos
cfg.RECORD_GUI = True
cfg.RESULT_DIR = Path(args.folder, "gui_eval")
cfg.IMG_FOLDER = Path(args.folder, "avideo_rotated.mp4")
cfg.LOOP_VIDEO = True
print(f"Processing {cfg.IMG_FOLDER}")
cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
app = QApplication.instance()  # https://stackoverflow.com/questions/46304070/multiple-qapplication-instances
if app is None:
    print("No QApplication instance found -> no other qt-based visualizer has been loaded")
    app = QApplication(sys.argv)
gui = AkevapVisual()

try:
    if cfg.NO_GUI:
        while True:
            if not gui.next_frame():
                break
    else:
        gui.main(app)
except Exception as e:
    print(e)
finally:
    import pandas as pd
    df = pd.DataFrame.from_dict(gui.CACHE, orient='index')
    df.to_pickle(Path(cfg.RESULT_DIR, 'gui_data.pkl'))

    if gui.cap.isOpened():
        gui.cap.release()
    if cfg.RECORD_GUI:
        gui.writer_gui.close()
    gui.writer_video.close()

    del gui
    print("DONE - kill process")
