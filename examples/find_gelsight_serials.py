import sys
import os

# Add the GelSight module path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gelmini_path = os.path.abspath(os.path.join(script_dir, "../VTLA_Data_Collect-main"))
if gelmini_path not in sys.path:
    sys.path.insert(0, gelmini_path)

# Debug: Print sys.path to verify the module path
print("sys.path:")
for path in sys.path:
    print(path)

try:
    from gelmini import gsdevice
except ModuleNotFoundError as e:
    print("Error: Unable to import 'gelmini'. Ensure the path is correct and the module is properly structured.")
    print(f"Details: {e}")
    sys.exit(1)

def find_gelsight_serials():
    """
    Query and print the serial numbers of connected GelSight devices using v4l2-ctl.
    """
    import subprocess
    import re

    print("Searching for GelSight devices...")
    devices = []
    
    # Check first 10 camera indices
    for index in range(10):
        try:
            # Run v4l2-ctl to get device info
            result = subprocess.run(['v4l2-ctl', '-d', str(index), '--info'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout
                # Check if it's a GelSight (Arducam or Mini)
                if 'Arducam' in output or 'Mini' in output:
                    print(f"\nDevice at index {index} (GelSight detected):")
                    print(output)
                    # Try to extract serial number from the output
                    # Look for "Serial : " followed by the serial
                    serial_match = re.search(r'Serial\s*:\s*(\S+)', output)
                    if serial_match:
                        serial = serial_match.group(1)
                        devices.append((index, serial))
                        print(f"Extracted Serial: {serial}")
                    else:
                        print("Could not extract serial number.")
        except subprocess.TimeoutExpired:
            continue
        except FileNotFoundError:
            print("v4l2-ctl not found. Please install v4l-utils.")
            return
        except Exception as e:
            continue
    
    if not devices:
        print("No GelSight devices found with extractable serial numbers.")
    else:
        print(f"\nTotal GelSight devices with serials found: {len(devices)}")
        for idx, serial in devices:
            print(f"  Camera index {idx}: Serial {serial}")

if __name__ == "__main__":
    find_gelsight_serials()
