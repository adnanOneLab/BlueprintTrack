import cv2

class BlueprintProcessor:
    def __init__(self):
        self.image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.origin = (0, 0)
        self.floor = 1
        self.cameras = {}
        self.stores = {}

    def load_blueprint(self, image_path):
        """Load the blueprint image from file"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for display
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print(f"Blueprint loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return True
    
    def set_scale(self, pixels, meters):
        """Set the scale factor (meters per pixel)"""
        self.scale_factor = meters / pixels
        print(f"Scale set: {self.scale_factor:.4f} meters per pixel")
        return True
    
    def set_origin(self, x, y):
        """Set the origin point (in pixels)"""
        self.origin = (x, y)
        print(f"Origin set: pixel coordinates ({x}, {y})")
        return True
    
    def set_floor(self, floor_number):
        """Set the floor number for the current blueprint"""
        self.floor = floor_number
        print(f"Floor set: {floor_number}")
        return True
        ...

    def pixels_to_meters(self, px, py):
        """Convert pixel coordinates to meter coordinates"""
        rel_x = px - self.origin[0]
        rel_y = self.origin[1] - py  # Invert y-axis
        
        meter_x = rel_x * self.scale_factor
        meter_y = rel_y * self.scale_factor
        
        return meter_x, meter_y
    
    def meters_to_pixels(self, mx, my):
        """Convert meter coordinates to pixel coordinates"""
        rel_x = mx / self.scale_factor
        rel_y = my / self.scale_factor
        
        px = int(self.origin[0] + rel_x)
        py = int(self.origin[1] - rel_y)  # Invert y-axis
        
        return px, py
