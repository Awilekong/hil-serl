import sys  # Ensure sys is imported
import os
# Add the GelSight module path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gelmini_path = os.path.abspath(os.path.join(script_dir, "../VTLA_Data_Collect-main"))
if gelmini_path not in sys.path:
    sys.path.insert(0, gelmini_path)

from gelmini import gsdevice  # Import GelSight SDK

# GelSight initialization
gelsight_1 = gsdevice.Camera(0)
gelsight_1.connect()
gelsight_2 = gsdevice.Camera(0)
gelsight_2.connect()