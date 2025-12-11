import cv2
import numpy as np
from collections import deque

class VisionController:
    """
    "Vision-based lane centering controller for steering correction"
    """

    def __init__(self, debug_graphs=False):
        self.stats = {
            'total_frames': 0,
            'vision_frames': 0,
            'rejected_frames': 0,
            'low_conf_frames': 0,
            'avg_lines_detected': []
        }
        self.debug_graphs = debug_graphs
        self.last_mu_adjustment = 1.0
        self.frame_count = 0

        self.curvature_reducer = deque(maxlen=5)
        self.confidence_reducer = deque(maxlen=5)

        self.adjustment_history = deque(maxlen=3)
        self.slope_history = deque(maxlen=10)
        self.max_adj_change = 0.15

    def get_mu_adjustment(self, camera_image, current_section, current_speed_kmh, current_waypoint_idx):
        """
        Analyze the camera image and return a mu adjustment factor for speed.
        Args: camera_image (np.ndarray): The input camera image from the vehicle
        Returns: float The mu adjustment factor for speed.
        """
        if camera_image is None:
            return 1.0

        if 6260 <= current_waypoint_idx <= 6290: #Sharp turn before long straight
            return 1.0
        
        self.stats['total_frames'] += 1
        self.stats['vision_frames'] += 1
        self.frame_count += 1
        #Every second frame
        #if self.frame_count % 2 != 0:
        #    return self.last_mu_adjustment
        
        #Track curvature analysis
        numpy_camera_img = np.array(camera_image)
        curvature, confidence = self.analyze_curvature(numpy_camera_img)

        self.curvature_reducer.append(curvature)
        self.confidence_reducer.append(confidence)

        avg_curvature = np.mean(self.curvature_reducer)
        avg_confidence = np.mean(self.confidence_reducer)

        if current_speed_kmh > 150: #Original 500, high
            avg_confidence *= 0.5
        elif current_speed_kmh > 100: #Orginal 300, low
            avg_confidence *= 0.7
    
        mu_adjustment = self.calculate_adj(avg_curvature, avg_confidence) #0.7 to 1.3 Original
        adj_deff = abs(mu_adjustment - self.last_mu_adjustment)
        if adj_deff > self.max_adj_change:
            if self.debug_graphs:
                print("Capping Mu Adjustment Change Rate")
            if mu_adjustment > self.last_mu_adjustment:
                mu_adjustment = self.last_mu_adjustment + self.max_adj_change
            else:
                mu_adjustment = self.last_mu_adjustment - self.max_adj_change


        smoothing = 0.7 #0.7 Original, reduces jitter and adjustments
        mu_adjustment = smoothing * self.last_mu_adjustment + (1 - smoothing) * mu_adjustment
        self.last_mu_adjustment = mu_adjustment

        return mu_adjustment
    
    def calculate_adj(self, curvature, confidence):
        """
        Calculate the mu adjustment based on curvature and confidence
        """
        if confidence < 0.3:
            self.stats['low_conf_frames'] += 1
            if self.debug_graphs:
                print("Uncertain")
                print(f"Vision Controller - Curvature: {curvature:.2f}, Confidence: {confidence:.2f}, Mu Adj: 1.01")
            return 1.005 #low confidence, no adjustment
        
        #Weighting factors
        #TODO: tune better thresholds, currently too linear. Improve curvature detection first
        if curvature >= 0.96:
            adj = 1.17
            if self.debug_graphs:
                print("Extremely Straight Road Adjustment")
        elif curvature >= 0.94:
            adj = 1.06
            if self.debug_graphs:
                print("Straight Road Adjustment")
        elif curvature >= 0.925:
            adj = 1.033
            if self.debug_graphs:
                print("Straight Road Adjustment")
        elif curvature >= 0.89:
            adj = 1.01
            if self.debug_graphs:
                print("Semi Straight Road Adjustment")
        elif curvature >= 0.87:
            adj = 1.00
            if self.debug_graphs:
                print("Semi Straight Road Adjustment")
        elif curvature >= 0.85:
            adj = 0.99
            if self.debug_graphs:
                print("Mild Curve Road Adjustment")
        elif curvature >= 0.79:
            adj = 0.97
            if self.debug_graphs:
                print("Tight Curve Road Adjustment")
        else:
            adj = 0.95
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
        - > 1 for straight roads
        - ~1 for small curves
        - <1 for very curvy roads
        Args: Img from camera
        """ 

        #Img mode, greyscale, contrast, and blur for edge detect
        grey_camera_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        grey_camera_img = clahe.apply(grey_camera_img)
        blurred_camera_img = cv2.GaussianBlur(grey_camera_img, (5, 5), 0)
        edges = cv2.Canny(blurred_camera_img, 50, 150)

        #Focusing on the visible portion of the track
        height, width = edges.shape
        far_top = int(height*0.15)
        far_left = int(width*0.25)
        far_right = int(width*0.75)
        near_bottom = int(height*0.36)
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
        lines = cv2.HoughLinesP(region_of_interest, 1, np.pi / 180, threshold=25, minLineLength=60, maxLineGap=50) #40, 113.5, 23
        #TODO: tune minLL and mLGap for better line detection
        #skipping some sections at curves

        if lines is None or len(lines) < 3:
            return 1.01, 0.0
        
        slopes = []
        line_len = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            
            #analyzing slope and position, removing near vertical and horizontal lines
            slope = abs(np.arctan((y2 - y1) / (x2 - x1)))
            deviation_difference = abs(slope - np.pi/2)
            line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if line_length < 35:
                continue

            if slope < 0.35:
                continue

            #TODO: Consider less restrictive slope thresholds
            if 0.6 < deviation_difference < 1.1: #0.5 and 1.1
                avg_horizontal = (x1 + x2) / 2
                avg_vert = (y1 + y2) / 2
                track_width = near_right - near_left

                if avg_vert < near_bottom - 20: #near bottom of roi
                    if 0.2*track_width < avg_horizontal-near_left < 0.8*track_width: #0.25 and 0.75
                        line_len.append(line_length)
                        slopes.append(deviation_difference)
        
        if not slopes or len(slopes) < 2:
            return 1.0, 0.0 #cant detect, assume all is normal
        
        avg_slope = np.mean(slopes)

        if len(self.slope_history) >= 3:
            new_avg = np.mean(self.slope_history)
            slope_change = abs(avg_slope - new_avg)
            if slope_change > 0.08:
                if self.debug_graphs:
                    print("Rejected")
                if len(self.curvature_reducer) > 0:
                    return self.curvature_reducer[-1], self.confidence_reducer[-1]
                return 1.0, 0.0
        self.slope_history.append(avg_slope)

        num_detected_lines_len = len(slopes)
        self.stats['avg_lines_detected'].append(num_detected_lines_len)
        avg_detected_line_len = np.mean(line_len)
        num_lines_confidence = min(1.0, num_detected_lines_len/8.0)#Original 10
        length_lines_confidence = min(1.0, avg_detected_line_len/80.0) #Original 100
        slope_consistency_confidence = max(0.0, 1.0 - np.std(slopes)*0.6) #Original 0.8
        #TODO: currrently too harsh on confidence, tune better
        confidence = (num_lines_confidence * 0.4 + length_lines_confidence * 0.25 + slope_consistency_confidence * 0.25)
        
        #curvature estimation
        #TODO: tune better thresholds
        if avg_slope > 0.94:
            if self.debug_graphs:
                print("Extremely Straight Road Detected")
            curvature = 1.2 #1.3 from 0.75
        elif avg_slope > 0.93:
            if self.debug_graphs:
                print("Very Straight Road Detected")
            curvature = 1.15 #1.15 from 0.8
        elif avg_slope > 0.92:
            if self.debug_graphs:
                print("Straight Road Detected")
            curvature = 1.05 #1.08 from 0.85
        elif avg_slope > 0.89:
            if self.debug_graphs:
                print("Moderate Curves Detected")
            curvature = 1.00 #1.03 from 0.9
        elif avg_slope > 0.87:
            if self.debug_graphs:
                print("Mild Curves Detected")
            curvature = 0.95 #1.0 from 0.95
        elif avg_slope > 0.85:
            if self.debug_graphs:
                print("Somewhat Tight Curves Detected")
            curvature = 0.92 #0.95 from 1.0
        elif avg_slope > 0.8:
            if self.debug_graphs:
                print("Somewhat Tight Curves Detected")
            curvature = 0.9 #0.95 from 1.0
        elif avg_slope > 0.77:
            if self.debug_graphs:
                print("Tight Curves Detected")
            curvature = 0.87 #0.93 from 1.05
        elif avg_slope > 0.75:
            if self.debug_graphs:
                print("Very Very Tight Curves Detected")
            curvature = 0.85 #0.9 from 1.1
        else:
            curvature = 0.825
        
        if self.debug_graphs:
            print("Extremely Tight Curves Detected")
            if self.debug_graphs:
                debug_img = cv2.cvtColor(region_of_interest, cv2.COLOR_RGB2BGR)
                cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if x2 - x1 == 0:
                            continue
                        slope = abs(np.arctan((y2 - y1) / (x2 - x1)))
                        deviation_difference = abs(slope - np.pi/2)
                        avg_x_horizontal = (x1 + x2) / 2
                        track_width = near_right - near_left
                        if deviation_difference > 0.7 and 0.25*track_width < avg_x_horizontal-near_left < 0.75*track_width:
                            cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if slopes:
                    avg_slope = np.mean(slopes)
                    cv2.putText(debug_img, f"Avg Slope: {avg_slope:.3f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(debug_img, f"Avg Curvature: {curvature:.3f}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(debug_img, f"Confidence: {confidence:.3f}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            
                cv2.imshow("Slope Debug", debug_img)
                cv2.waitKey(1)

        return curvature, confidence
    
    def get_information(self):
        if self.stats['total_frames'] == 0:
            return "no information available"
        
        avg_lines = np.mean(self.stats['avg_lines_detected']) if self.stats['avg_lines_detected'] else 0
        reject_rate = self.stats['rejected_frames'] / self.stats['total_frames'] * 100
        low_conf_rate = self.stats['low_conf_frames'] / self.stats['total_frames'] * 100
        info = (f"Info:\n" f"Avg Lines Detected: {avg_lines:.2f}\n" f"Rejected Frms: {self.stats['rejected_frames']} ({reject_rate:.2f}%)\n" f"Low Confidence Frms: {self.stats['low_conf_frames']} ({low_conf_rate:.2f}%)\n")
        return info
    
    def reset_stats(self):
        self.stats = {
            'total_frames': 0,
            'vision_frames': 0,
            'rejected_frames': 0,
            'low_conf_frames': 0,
            'avg_lines_detected': []
        }
        