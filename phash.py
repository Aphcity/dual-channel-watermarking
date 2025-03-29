import numpy as np
import cv2
from PIL import Image
def calculate_phash(image_paths, hash_size=8):
    """
    Calculate the perceptual hash (pHash) of one or more images.

    Args:
        image_paths (str or list): Path to the input image or a list of image paths.
        hash_size (int): Size of the hash (default is 8x8).

    Returns:
        str or list: The perceptual hash as a hexadecimal string for a single image,
                     or a list of hexadecimal strings for multiple images.
    """
    def process_image(image_path):
        # Step 1: Load the image and convert it to grayscale
        image = Image.open(image_path).convert("L")
        
        # Step 2: Resize the image to (hash_size * 4, hash_size * 4) for better DCT accuracy
        resized_image = image.resize((hash_size * 4, hash_size * 4), Image.ANTIALIAS)
        
        # Step 3: Convert the image to a numpy array
        pixel_array = np.array(resized_image, dtype=np.float32)
        
        # Step 4: Apply the Discrete Cosine Transform (DCT)
        dct = cv2.dct(pixel_array)
        
        # Step 5: Keep only the top-left (hash_size x hash_size) part of the DCT
        dct_low_freq = dct[:hash_size, :hash_size]
        
        # Step 6: Compute the median value of the DCT coefficients (excluding the DC coefficient)
        median_value = np.median(dct_low_freq[1:])
        
        # Step 7: Generate the hash: 1 if the DCT coefficient is above the median, else 0
        hash_bits = dct_low_freq > median_value
        hash_bits = hash_bits.flatten()
        
        # Step 8: Convert the binary hash to a hexadecimal string
        hash_hex = ''.join(['1' if bit else '0' for bit in hash_bits])
        hash_hex = hex(int(hash_hex, 2))[2:].zfill(hash_size * hash_size // 4)
        
        return hash_hex

    # Check if the input is a single image path or a list of paths
    if isinstance(image_paths, str):
        return process_image(image_paths)
    elif isinstance(image_paths, list):
        return [process_image(image_path) for image_path in image_paths]
    else:
        raise ValueError("image_paths must be a string or a list of strings.")

# Example usage
if __name__ == "__main__":
    # Single image
    single_image_path = "example.jpg"
    phash = calculate_phash(single_image_path)
    print(f"Perceptual Hash (single image): {phash}")
    
    # Batch of images
    batch_image_paths = ["example1.jpg", "example2.jpg", "example3.jpg"]
    phashes = calculate_phash(batch_image_paths)
    print(f"Perceptual Hashes (batch): {phashes}")