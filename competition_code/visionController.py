import cv2
import numpy as np
from collections import deque

class VisionController:
    """
    "Vision-based lane centering controller for steering correction"
    """

    def __init__(self, debug_graphs=False):
        self.debug_graphs = debug_graphs
        self.last_mu_adjustment = 1.0
        self.frame_count = 0

        self.curvature_reducer = deque(maxlen=5)
        self.confidence_reducer = deque(maxlen=5)
        self.vision_disabler = [ #certain curves and regions better left untouched
        ]

        self.vision_disabler_sections = [] #use waypoint instead
    
    def get_mu_adjustment(self, camera_image, current_section, current_speed_kmh, current_waypoint_idx):
        """
        Analyze the camera image and return a mu adjustment factor for speed.
        Args: camera_image (np.ndarray): The input camera image from the vehicle
        Returns: float The mu adjustment factor for speed.
        """
        if camera_image is None:
            return 1.0
        
        self.frame_count += 1

        if self.should_disable_vision(current_section, current_waypoint_idx):
            return 1.0
        
        #Every second frame
        #if self.frame_count % 2 != 0:
        #    return self.last_mu_adjustment
        
        #Track curvature and width analysis
        numpy_camera_img = np.array(camera_image)
        curvature, confidence = self.analyze_curvature(numpy_camera_img)

        self.curvature_reducer.append(curvature)
        self.confidence_reducer.append(confidence)

        avg_curvature = np.mean(self.curvature_reducer)
        avg_confidence = np.mean(self.confidence_reducer)

        if current_speed_kmh > 150: #Original 500, high
            avg_curvature *= 0.5
        elif current_speed_kmh > 100: #Orginal 300, low
            avg_confidence *= 0.7
    
        mu_adjustment = self.calculate_adj(avg_curvature, avg_confidence) #0.7 to 1.3 Original
        smoothing = 0.5 #0.7 Original, reduces jitter and adjustments
        mu_adjustment = smoothing * self.last_mu_adjustment + (1 - smoothing) * mu_adjustment
        self.last_mu_adjustment = mu_adjustment

        return mu_adjustment
    
    def should_disable_vision(self, current_section, current_waypoint_idx):
        """
        Determine if vision-based adjustments should be disabled
        """

        if current_section in self.vision_disabler_sections:
            return True
        
        for start, end in self.vision_disabler:
            if start <= current_waypoint_idx <= end:
                return True

        return False
    
    def calculate_adj(self, curvature, confidence):
        """
        Calculate the mu adjustment based on curvature and confidence
        """
        if confidence < 0.3:
            print("Uncertain")
            print(f"Vision Controller - Curvature: {curvature:.2f}, Confidence: {confidence:.2f}, Mu Adj: 1.05")
            return 1.02 #low confidence, no adjustment
        
        #Weighting factors
        if curvature >= 0.95:
            adj = 1.3
            if self.debug_graphs:
                print("Extremely Straight Road Adjustment")
        elif curvature >= 0.9:
            adj = 1.2
            if self.debug_graphs:
                print("Straight Road Adjustment")
        elif curvature >= 0.85:
            adj = 1.15
            if self.debug_graphs:
                print("Semi Straight Road Adjustment")
        elif curvature >= 0.79:
            adj = 1.1
            if self.debug_graphs:
                print("Mild Curve Road Adjustment")
        elif curvature >= 0.7:
            adj = 1.00
            if self.debug_graphs:
                print("Tight Curve Road Adjustment")
        else:
            adj = 0.98
            if self.debug_graphs:
                print("Extremely Curvy Road Adjustment")
        
        mu_adjustment = 1.0 + (adj-1.0) * confidence
        mu_adjustment = max(0.98, min(1.3, mu_adjustment)) #Orginial 0.8 to 1.3
        if self.debug_graphs:
            print(f"Vision Controller - Curvature: {curvature:.2f}, Confidence: {confidence:.2f}, Mu Adj: {mu_adjustment:.2f}")
        return mu_adjustment

    def analyze_curvature(self, image):
        """
        Analyze the curvature of the road in the image.
        
        Returns: Curvature factor
            1 > straight road, open space
            1 < tight curves, narrow space
            ~1 normal, moderate curves
        """ 
        #Processing images to detect edges and lines
        grey_camera_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #Increase contrast of the image through CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        grey_camera_img = clahe.apply(grey_camera_img)
        #Gaussian Blur to reduce noise
        blurred_camera_img = cv2.GaussianBlur(grey_camera_img, (3, 3), 0)
        #Edge detection through Canny
        edges = cv2.Canny(blurred_camera_img, 50, 150)

        #Focusing on the visible portion of the track
        height, width = edges.shape
        far_top = int(height*0.05)
        far_left = int(width*0.35)
        far_right = int(width*0.65)
        near_bottom = int(height*0.55)
        near_left = int(width*0.2)
        near_right = int(width*0.8)
        trapezoid = np.zeros_like(edges)
        pts = np.array([[far_left, near_bottom],
                        [far_right, near_bottom],
                        [near_right, far_top],
                        [near_left, far_top]],
                        np.int32)
        
        cv2.fillPoly(trapezoid, [pts], 255)
        region_of_interest = cv2.bitwise_and(edges, trapezoid)

        #Hough Transform to detect lines
        lines = cv2.HoughLinesP(region_of_interest, 1, np.pi / 180, threshold=40, minLineLength=113.5, maxLineGap=23) #50, 115.5, 21
        #Edit for amt of lines, sensitivity, and connection (Orginal values: 40, 50, 150)

        if lines is None or len(lines) < 3:
            return 1.01, 0.0
        
        slopes = []
        line_len = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue

            slope = abs(np.arctan((y2 - y1) / (x2 - x1))) #How far it is from vertical path
            deviation_difference = abs(slope - np.pi/2)
            line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if 0.7 < deviation_difference < 1.0: #filtering out near vertical and near horizontal lines
                avg_horizontal = (x1 + x2) / 2
                track_width = near_right - near_left

                if 0.25*track_width < avg_horizontal < 0.75*track_width: #reasonable horizontal pos
                    line_len.append(line_length)
                    slopes.append(deviation_difference)

        if self.debug_graphs:
            debug_img = cv2.cvtColor(region_of_interest, cv2.COLOR_RGB2BGR)
            cv2.polylines(debug_img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if slopes:
                avg_slope = np.mean(slopes)
                cv2.putText(debug_img, f"Avg Slope: {avg_slope:.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            
            cv2.imshow("Slope Debug", debug_img)
            cv2.imshow("Curvature Debug", debug_img)
            cv2.waitKey(1)
        
        if not slopes or len(slopes) < 2:
            return 1.01, 0.0 #cant detect, assume all is normal
        
        num_detected_lines_len = len(slopes)
        avg_detected_line_len = np.mean(line_len)
        num_lines_confidence = min(1.0, num_detected_lines_len/10.0)#Original 10
        length_lines_confidence = min(1.0, avg_detected_line_len/100.0) #Original 100
        slope_consistency_confidence = max(0.0, 1.0 - np.std(slopes)*0.8) #Original 0.8

        confidence = (num_lines_confidence * 0.4 + length_lines_confidence * 0.3 + slope_consistency_confidence * 0.3 )
        avg_slope = np.mean(slopes)
        
        if avg_slope < 0.75:
            if self.debug_graphs:
                print("Extremely Straight Road Detected")
            curvature = 1.3 #extremely straight road
        elif avg_slope < 0.8:
            if self.debug_graphs:
                print("Very Straight Road Detected")
            curvature = 1.15 #very straight road
        elif avg_slope < 0.85:
            if self.debug_graphs:
                print("Straight Road Detected")
            curvature = 1.08 #straight road
        elif avg_slope < 0.9:
            if self.debug_graphs:
                print("Moderate Curves Detected")
            curvature = 1.03 #moderate curves
        elif avg_slope < 0.95:
            if self.debug_graphs:
                print("Mild Curves Detected")
            curvature = 1.0 #mild curves
        elif avg_slope < 1.0:
            if self.debug_graphs:
                print("Somewhat Tight Curves Detected")
            curvature = 0.95 #somewhat tight curves
        elif avg_slope < 1.05:
            if self.debug_graphs:
                print("Tight Curves Detected")
            curvature = 0.93 #tight curves
        elif avg_slope < 1.1:
            if self.debug_graphs:
                print("Very Very Tight Curves Detected")
            curvature = 0.9 #very very tight curves
        else:
            curvature = 0.87
            if self.debug_graphs:
                print("Extremely Tight Curves Detected")

        return curvature, confidence

