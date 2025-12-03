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
            (500, 570)
        ]

    
    def get_mu_adjustment(self, camera_image, current_speed_kmh, current_waypoint_idx):
        """
        Analyze the camera image and return a mu adjustment factor for speed.
        Args: camera_image (np.ndarray): The input camera image from the vehicle
        Returns: float The mu adjustment factor for speed.
            1.0 = no changes made
            > 1.0 = speed can be increased (large open space, small curves, wide area)
            < 1.0 = speed should be decreased (tight curves, narrow area)
        """
        if camera_image is None:
            return 1.0
        
        self.frame_count += 1

        if self._should_disable_vision():
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

        if current_speed_kmh > 500: #Original 500, high
            avg_curvature *= 0.5 #Caution at high speeds
        elif current_speed_kmh > 300: #Orginal 300, low
            avg_confidence *= 0.7
    
        #Limiting extreme adjustments
        mu_adjustment = np.clip(mu_adjustment, 0.8, 1.35) #0.7 to 1.3 Original
        
        smoothing = 0.8 #0.7 Original, reduces jitter and adjustments
        mu_adjustment = smoothing * self.last_mu_adjustment + (1 - smoothing) * mu_adjustment
        self.last_mu_adjustment = mu_adjustment

        if self.debug_graphs:
            print(f"Vision Controller - Curvature: {curvature:.2f}, Mu Adjustment: {mu_adjustment:.2f}")
            #Track Width: {track_width:.2f}
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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        grey_camera_img = clahe.apply(grey_camera_img)
        #Gaussian Blur to reduce noise
        blurred_camera_img = cv2.GaussianBlur(grey_camera_img, (9, 9), 0)
        #Edge detection through Canny
        edges = cv2.Canny(blurred_camera_img, 50, 150)

        #Focusing on the visible portion of the track
        height, width = edges.shape
        top = int(height*0.3)
        bottom = int(height*0.55)
        region_of_interest = edges[top:bottom, :]

        #Hough Transform to detect lines
        lines = cv2.HoughLinesP(region_of_interest, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=150)

        if lines is None or len(lines) < 2:
            return 1.05 #cant detect, assume all is normal
        
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = abs(np.arctan((y2 - y1) / (x2 - x1))) #How far it is from vertical path
            deviation_difference = abs(slope-np.pi/2)
            if 0.2 < deviation_difference < 1.5: #filtering out near vertical and near horizontal lines
                slopes.append(deviation_difference)

        if self.debug_graphs:
            debug_img = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Cervature Debug", debug_img)
            cv2.waitKey(1)
        
        if not slopes or len(slopes) < 2:
            return 1.05 #cant detect, assume all is normal
        
        avg_slope = np.mean(slopes)
        if avg_slope < 1.05:
            print("Extremely Straight Road Detected")
            return 1.3 #extremely straight road
        elif avg_slope < 1.1:
            print("Very Straight Road Detected")
            return 1.25 #very straight road
        elif avg_slope < 1.15:
            print("Straight Road Detected")
            return 1.2 #straight road
        elif avg_slope < 1.20:
            print("Moderate Curves Detected")
            return 1.1 #moderate curves
        elif avg_slope < 1.21:
            print("Mild Curves Detected")
            return 1.0 #mild curves
        elif avg_slope < 1.22:
            print("Somewhat Tight Curves Detected")
            return 0.9 #somewhat tight curves
        elif avg_slope < 1.23:
            print("Tight Curves Detected")
            return 0.85 #tight curves
        elif avg_slope < 1.24:
            print("Very Very Tight Curves Detected")
            return 0.825 #very very tight curves
        elif avg_slope < 1.25:
            print("Extremely Tight Curves Detected")
            return 0.8#extremely tight curves
        else:
            return 0.8

    def visualize(self, image, mu_adjustment):
        """
        Debuggging Purposes to see what the camer is detecting
        Call after mu_adjustment calculation to see the effect
        """
        if image is None:
            return
        
        numpy_camera_img = np.array(image)
        modified_img = numpy_camera_img.copy()

        #depicting mu adjustment on the image
        height, width = modified_img.shape[:2]

        if mu_adjustment >= 1.0:
            color = (0, 255, 0) #Green for open space : Faster
        else:
            color = (255, 0, 0) #Red for tight space : Slower
        
        cv2.putText(modified_img, f"Mu Adj: {mu_adjustment:.2f}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        top = int(height * 0.25)
        bottom = int(height * 0.60)
        cv2.rectangle(modified_img, (0, top), (width, bottom), color, 2)

        cv2.imshow("Vision Controller Debug", modified_img)
        cv2.waitKey(1)
        


