import cv2
import numpy as np

class VisionController:
    """
    "Vision-based lane centering controller for steering correction"
    """

    def __init__(self, debug_graphs=False):
        self.debug_graphs = debug_graphs
        self.last_mu_adjustment = 1.0
        self.frame_count = 0
    
    def get_mu_adjustment(self, camera_image, current_speed_kmh):
        """
        Analyze the camera image and return a mu adjustment factor for steering.
        Args: camera_image (np.ndarray): The input camera image from the vehicle AND current_speed_kmh (float): The current speed of the vehicle in km/h.
        Returns: float The mu adjustment factor for steering.
            1.0 = no changes made
            > 1.0 = speed can be increased (large open space, small curves, wide area)
            < 1.0 = speed should be decreased (tight curves, narrow area)
        """
        if camera_image is None:
            return self.last_mu_adjustment
        
        self.frame_count += 1
        
        #Every third frame
        if self.frame_count % 2 != 0:
            return self.last_mu_adjustment
        
        #Track curvature and width analysis
        numpy_camera_img = np.array(camera_image)
        curvature = self._analyze_curvature(numpy_camera_img)
        #track_width = self._analyze_road_width(numpy_camera_img)

        #Track width and curvature 
        # if curvature > 1.0 and track_width > 1.0:
        #     mu_adjustment = (curvature + track_width) / 2.0 #Open Space
        # elif curvature < 1.0 or track_width < 1.0:
        #     mu_adjustment = min(curvature, track_width) #Tight Space
        # else:
        #     mu_adjustment = 0.7 * 1.0 + 0.15 * curvature + 0.15 * track_width #Contradictory, Normal Space
        
        #Curvature Only
        mu_adjustment = curvature
    
        #Limiting extreme adjustments
        mu_adjustment = np.clip(mu_adjustment, 0.95, 2.0) #0.7 to 1.3 Original
        
        smoothing = 0.3 #0.7 Original, reduces jitter and adjustments
        mu_adjustment = smoothing * self.last_mu_adjustment + (1 - smoothing) * mu_adjustment
        self.last_mu_adjustment = mu_adjustment

        if self.debug_graphs:
            print(f"Vision Controller - Curvature: {curvature:.2f}, Mu Adjustment: {mu_adjustment:.2f}")
            #Track Width: {track_width:.2f}

        return mu_adjustment

    def _analyze_curvature(self, image):
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
        edges = cv2.Canny(blurred_camera_img, 150, 200)

        #Focusing on the visible portion of the track
        height, width = edges.shape
        top = int(height*0.30)
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
        if avg_slope < 1.15:
            print("Very Straight Road Detected")
            return 2.0 #very straight road
        elif avg_slope < 1.18:
            print("Straight Road Detected")
            return 1.9 #straight road
        elif avg_slope < 1.2:
            print("Moderate Curves Detected")
            return 1.75 #moderate curves
        elif avg_slope < 1.225:
            print("Mild Curves Detected")
            return 1.6 #mild curves
        elif avg_slope < 1.25:
            print("Somewhat Tight Curves Detected")
            return 1.4 #somewhat tight curves
        elif avg_slope < 1.3:
            print("Tight Curves Detected")
            return 1.3 #tight curves
        elif avg_slope < 1.35:
            print("Very Tight Curves Detected")
            return 1.2 #very tight curves
        elif avg_slope < 1.38:
            print("Very Very Tight Curves Detected")
            return 1.1 #very tight curves
        elif avg_slope < 1.4:
            print("Extremely Tight Curves Detected")
            return 1.00 #very tight curves
        else:
            return 0.95 #very tight curves
    
    # def _analyze_road_width(self, image):
    #     """
    #     Analyze the width of the road directly in front of the vehicle.
        
    #     Return: Width
    #         1.00 > wide road, open space
    #         1.00 < narrow road, tight space
    #         ~1.00  normal width
    #     """

    #     grey_camera_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     #Increase contrast of the image through CLAHE
    #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    #     grey_camera_img = clahe.apply(grey_camera_img)
    #     #Gaussian Blur to reduce noise
    #     blurred_camera_img = cv2.GaussianBlur(grey_camera_img, (9, 9), 0)
    #     #Edge detection through Canny
    #     edges = cv2.Canny(blurred_camera_img, 120, 200)

    #     #Focusing on the visible portion of the track
    #     height, width = edges.shape
    #     mid_top = int(height*0.4)
    #     mid_bottom = int(height*0.6)
    #     region_of_interest = edges[mid_top:mid_bottom, :]

    #     #Detect left and right edges (indicative  of road border)
    #     horizontal_sum = np.sum(region_of_interest, axis=0)
    #     threshold = np.sum(horizontal_sum) * 0.3
    #     edge_pos = np.where(horizontal_sum > threshold)[0]

    #     if self.debug_graphs:
    #         debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #         cv2.imshow("Road Width Debug", debug_img)
    #         cv2.waitKey(1)

    #     if(len(edge_pos) < 2):
    #         return 1.0 #cant detect, assume all is normal
        
    #     left_edge = edge_pos[0]
    #     right_edge = edge_pos[-1]
    #     estimated_width = right_edge - left_edge
    #     #lane to image width ratio
    #     width_ratio = estimated_width / width

    #     if width_ratio > 0.65:
    #         return 1.2  # very wide region
    #     elif width_ratio < 0.55:
    #         return 1.1 # semi wide region
    #     elif width_ratio < 0.45:
    #         return 1.05 # normal width
    #     elif width_ratio < 0.35:
    #         return 0.975 # semi narrow region
    #     else:
    #         return 0.95 # very narrow region

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

        if mu_adjustment >= 1.25:
            color = (0, 255, 0) #Green for open space : Faster
        else:
            color = (0, 165, 255) #Red for tight space : Slower
        
        cv2.putText(modified_img, f"Mu Adj: {mu_adjustment:.2f}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        top = int(height * 0.25)
        bottom = int(height * 0.60)
        cv2.rectangle(modified_img, (0, top), (width, bottom), color, 2)

        cv2.imshow("Vision Controller Debug", modified_img)
        cv2.waitKey(1)
        


