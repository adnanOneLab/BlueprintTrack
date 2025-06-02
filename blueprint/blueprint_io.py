import json

class BlueprintIOMixin:
    def export_to_json(self, output_file):
        """Export the spatial data to a JSON file"""
        data = {
            "mall_info": {
                "name": "Mall Blueprint",
                "floors": [self.floor],
                "scale_factor": self.scale_factor,
                "origin": {"x": self.origin[0], "y": self.origin[1]}
            },
            "cameras": list(self.cameras.values()),
            "stores": list(self.stores.values()),
            "reference_points": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {output_file}")
        return True
    
    def import_from_json(self, input_file):
        """Import spatial data from a JSON file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Load mall info
        if "mall_info" in data:
            mall_info = data["mall_info"]
            if "scale_factor" in mall_info:
                self.scale_factor = mall_info["scale_factor"]
            if "origin" in mall_info:
                self.origin = (mall_info["origin"]["x"], mall_info["origin"]["y"])
            if "floors" in mall_info and len(mall_info["floors"]) > 0:
                self.floor = mall_info["floors"][0]
        
        # Load cameras and stores
        if "cameras" in data:
            self.cameras = {camera["camera_id"]: camera for camera in data["cameras"]}
        if "stores" in data:
            self.stores = {store["store_id"]: store for store in data["stores"]}
        
        print(f"Data imported from {input_file}")
        print(f"Loaded: {len(self.cameras)} cameras, {len(self.stores)} stores")
        return True

    def get_camera_count(self):
        """Get the current number of cameras"""
        return len(self.cameras)
    
    def get_store_count(self):
        """Get the current number of stores"""
        return len(self.stores)
    
    def clear_cameras(self):
        """Remove all cameras"""
        count = len(self.cameras)
        self.cameras.clear()
        print(f"Cleared {count} cameras")
        return True
    
    def clear_stores(self):
        """Remove all stores"""
        count = len(self.stores)
        self.stores.clear()
        print(f"Cleared {count} stores")
        return True
    
    def remove_camera(self, camera_id):
        """Remove a specific camera"""
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            print(f"Camera {camera_id} removed")
            return True
        return False
    
    def remove_store(self, store_id):
        """Remove a specific store"""
        if store_id in self.stores:
            del self.stores[store_id]
            print(f"Store {store_id} removed")
            return True
        return False