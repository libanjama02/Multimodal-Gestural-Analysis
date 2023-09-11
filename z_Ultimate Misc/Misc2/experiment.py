import os
import subprocess
import time

gesture_name = input("Enter gesture name: ")

# WIP:  Create directory to store data (should be in DataCollected)
data_dir = f"data/{gesture_name}"
os.makedirs(data_dir, exist_ok=True)

# Start data recording scripts 
processes = []
for script in ["imu_recording.py", "handpose_recording.py"]: 
    p = subprocess.Popen(["python3", script, "-f", f"{data_dir}/{script.split('.')[0]}.txt"], stdout=subprocess.PIPE)
    processes.append(p)

# Check if processes are alive, and if not, terminate all
while True:
    for p in processes:
        if p.poll() is not None:  # If the process has ended
            print(f"Error: {p.args[1]} failed.")
            for p_ in processes:
                p_.terminate()
            exit(1)
    time.sleep(1)

# Wait for user to start recording
input("Press Enter to start recording...")
# WIP code for recording goes here (needs to detect Enter as start)

# Wait for to stop recording
while True:
    answer = input("Stop recording (y/n)? ")
    if answer.lower() == 'y':
        # WIP code here to stop recording
        break

# WIP: Insert cleanup code here

print("Finished")
