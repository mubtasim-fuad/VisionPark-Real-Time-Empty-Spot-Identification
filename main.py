import cv2

from util import get_parking_spots_bboxes   # Import function to get bounding boxes of parking spots from connected components
from util import empty_or_not               # Import function to check if a parking spot is empty or not
import numpy as np
from matplotlib import pyplot as plt

# Function to calculate difference between two images based on mean pixel intensity
def calc_diffs(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Path to the mask image that highlights parking spots area
mask = './mask_1920_1080.png'

# Path to the video file showing parking lot footage
video_path = './data/parking_1920_1080_loop.mp4'

# Read the mask image in grayscale mode (0 flag)
mask = cv2.imread(mask, 0)

# Find connected components in the mask image; returns labels and stats for each connected region
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Extract bounding boxes of parking spots from connected components data using the helper function
spots = get_parking_spots_bboxes(connected_components)

# Initialize list to keep track of whether each parking spot is empty (True) or occupied (False)
spots_status = [None for j in spots]

# Initialize list to hold difference values between frames for each spot (used for detecting changes)
diffs = [None for j in spots]

# Variable to store the previous video frame for comparison with the current frame
previous_frame = None

# Open the video file for reading frames
cap = cv2.VideoCapture(video_path)

ret = True      # Flag to indicate if frame reading was successful

step = 30       # Number of frames to skip between analyses (process every 30th frame)

frame_nmr = 0   # Counter for frame number processed

# Main loop to read and process video frames
while ret:
    ret, frame = cap.read()   # Read a frame from the video

    # If current frame number is multiple of step and previous frame exists, compute difference for each spot
    if frame_nmr % step ==  0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            # Crop the current spot area from current frame
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            # Calculate difference with corresponding spot in previous frame
            diffs[spot_indx] = calc_diffs(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    # Every step frames, update the parking spot status (empty or not)
    if frame_nmr % step ==  0:
        # If this is the first frame, check all spots
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            # Otherwise, select spots with significant change (diff > 40% of max difference)
            arr_ = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs)>0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            # Crop the spot area in the current frame
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            # Determine if spot is empty or occupied
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    # Update the previous_frame variable to the current frame copy for next iteration comparison
    if frame_nmr % step ==  0:
        previous_frame = frame.copy()

    # For every spot, draw a rectangle on the frame with color depending on spot status
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]
        if spot_status:    # If spot is empty, draw green rectangle
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:              # If occupied, draw red rectangle
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Draw a black filled rectangle on frame for the info panel background
    cv2.rectangle(frame, (80, 20), (630, 110), (0,0,0), -1)

    # Put text on the frame showing how many spots are empty out of total spots
    cv2.putText(frame, "Available Empty Spots: {}/{}".format(str(sum(spots_status)), str(len(spots_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Put instruction text for quitting
    cv2.putText(frame, "Press Q to exit", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Create a window that can be resized
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # Display the frame with drawn rectangles and text
    cv2.imshow('frame',frame)

    # Wait 25ms for key press; if 'q' or 'Q' is pressed, exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1   # Increment frame counter

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
