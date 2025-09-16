# Yatin Chandar and Arun Showry

import numpy as np
import cv2

# Global variables
selected_hsv = None
tolerance_h = 10  # Hue tolerance
tolerance_s = 50  # Saturation tolerance  
tolerance_v = 50  # Value tolerance
current_frame = None
current_hsv = None
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global selected_hsv, current_frame, current_hsv, mouse_x, mouse_y
    
    if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
        # Ensure coordinates are within bounds
        height, width = current_hsv.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            # Get the HSV value of the clicked pixel
            clicked_hsv = current_hsv[y, x]
            selected_hsv = clicked_hsv.copy()
            mouse_x, mouse_y = x, y
            
            print(f"Selected pixel at ({x}, {y})")
            print(f"HSV values: H={clicked_hsv[0]}, S={clicked_hsv[1]}, V={clicked_hsv[2]}")

def create_hsv_mask(hsv_image, target_hsv, tol_h, tol_s, tol_v):
    """Create a mask based on HSV similarity to target color"""
    if target_hsv is None:
        return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    h_target = int(target_hsv[0])
    s_target = int(target_hsv[1])
    v_target = int(target_hsv[2])

    # Define lower and upper bounds for each channel
    # Hue wraps around at 180 (in OpenCV HSV)
    if h_target - tol_h < 0:
        # Hue wraps around low end
        lower1 = np.array([0, max(0, s_target - tol_s), max(0, v_target - tol_v)])
        upper1 = np.array([h_target + tol_h, min(255, s_target + tol_s), min(255, v_target + tol_v)])
        lower2 = np.array([180 + (h_target - tol_h), max(0, s_target - tol_s), max(0, v_target - tol_v)])
        upper2 = np.array([180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif h_target + tol_h > 180:
        # Hue wraps around high end
        lower1 = np.array([h_target - tol_h, max(0, s_target - tol_s), max(0, v_target - tol_v)])
        upper1 = np.array([180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
        lower2 = np.array([0, max(0, s_target - tol_s), max(0, v_target - tol_v)])
        upper2 = np.array([(h_target + tol_h) - 180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # No hue wraparound
        lower = np.array([h_target - tol_h, max(0, s_target - tol_s), max(0, v_target - tol_v)])
        upper = np.array([h_target + tol_h, min(255, s_target + tol_s), min(255, v_target + tol_v)])
        mask = cv2.inRange(hsv_image, lower, upper)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create windows and set mouse callback
cv2.namedWindow('Original')
cv2.namedWindow('Mask')
cv2.namedWindow('Result')
cv2.setMouseCallback('Original', mouse_callback)

# Create trackbars for tolerance adjustment
cv2.createTrackbar('Hue Tolerance', 'Mask', tolerance_h, 90, lambda x: None)
cv2.createTrackbar('Sat Tolerance', 'Mask', tolerance_s, 100, lambda x: None)
cv2.createTrackbar('Val Tolerance', 'Mask', tolerance_v, 100, lambda x: None)

print("\n=== Color Detection Tool ===")
print("Instructions:")
print("- Click on any pixel in the 'Original' window to select a color")
print("- Adjust tolerance trackbars to fine-tune detection")
print("- Green circle marks your selection point")
print("- Yellow contours show detected regions")
print("- Press 'r' to reset selection")
print("- Press 's' to save current mask")
print("- Press 'q' to quit")
print("============================\n")

frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break
    
    frame_count += 1
    
    # Flip frame horizontally for mirror effect (optional)
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV
    current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a display frame
    display_frame = frame.copy()
    
    # Get current tolerance values from trackbars
    tolerance_h = cv2.getTrackbarPos('Hue Tolerance', 'Mask')
    tolerance_s = cv2.getTrackbarPos('Sat Tolerance', 'Mask')
    tolerance_v = cv2.getTrackbarPos('Val Tolerance', 'Mask')
    
    # Create mask based on selected HSV value
    if selected_hsv is not None:
        mask = create_hsv_mask(current_hsv, selected_hsv, tolerance_h, tolerance_s, tolerance_v)
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on display frame
        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw all contours
            cv2.drawContours(display_frame, contours, -1, (0, 255, 255), 2)
            
            # Draw bounding box around largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate and draw centroid of largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(display_frame, (cx, cy), 7, (255, 0, 0), -1)
                cv2.circle(display_frame, (cx, cy), 9, (255, 255, 255), 2)
        
        # Mark selected pixel
        if mouse_x >= 0 and mouse_y >= 0:
            cv2.circle(display_frame, (mouse_x, mouse_y), 5, (0, 255, 0), -1)
            cv2.circle(display_frame, (mouse_x, mouse_y), 7, (0, 0, 0), 1)
        
        # Add text overlays
        cv2.putText(display_frame, f"Selected HSV: H={selected_hsv[0]} S={selected_hsv[1]} V={selected_hsv[2]}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Tolerances: H={tolerance_h} S={tolerance_s} V={tolerance_v}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(contours) > 0:
            cv2.putText(display_frame, f"Contours found: {len(contours)}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        # No color selected yet
        mask = np.zeros(current_hsv.shape[:2], dtype=np.uint8)
        result = np.zeros_like(frame)
        cv2.putText(display_frame, "Click on a pixel to select color", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Store current frame for mouse callback
    current_frame = display_frame
    
    # Display the frames
    cv2.imshow('Original', display_frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset selection
        selected_hsv = None
        mouse_x, mouse_y = -1, -1
        print("Selection reset")

# Cleanup
cap.release()
cv2.destroyAllWindows()