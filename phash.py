import hashlib

import PIL
import numpy as np
import cv2
import torch
from PIL import Image
import time
import os
def calculate_phash(image_paths, hash_size=8, path=True):
    """
    Calculate the perceptual hash (pHash) of one or more images.

    Args:
        image_paths (str or list): Path to the input image or a list of image paths.
        hash_size (int): Size of the hash (default is 8x8).

    Returns:
        str or list: The perceptual hash as a hexadecimal string for a single image,
                     or a list of hexadecimal strings for multiple images.
    """
    def process_image(image_path, istensor=False):
        # Step 1: Load the image and convert it to grayscale
        if istensor:
            image = image_path.convert("L")
        else:
            image = Image.open(image_path).convert("L")

        # Step 2: Resize the image to (hash_size * 4, hash_size * 4) for better DCT accuracy
        resized_image = image.resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        
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
    # if image_paths is a tensor
    elif isinstance(image_paths, list) and all(isinstance(image, PIL.Image.Image) for image in image_paths):
        return [process_image(image, istensor=True) for image in image_paths]
    elif isinstance(image_paths, list):
        return [process_image(image_path) for image_path in image_paths]
    else:
        raise ValueError("image_paths must be a string or a list of strings.")

def Hamming_distance(hash1,hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

# Example usage
if __name__ == "__main__":
    # Single image
    # single_image_path = "example.jpg"
    # phash = calculate_phash(single_image_path)
    # print(f"Perceptual Hash (single image): {phash}")
    
    # Batch of images
    batch_image_paths = ["test phash/im1.jpg", "test phash/im1_blur.jpg", "test phash/im1_text.jpg", "test phash/im1_crop.jpg", "test phash/im1_dot.jpg"]
    phashes = calculate_phash(batch_image_paths)
    # calculate the sha3 hash of the image
    sha3_hash = [hashlib.sha3_256(open(batch_image_paths[0], 'rb').read()).hexdigest(),
                 hashlib.sha3_256(open(batch_image_paths[1], 'rb').read()).hexdigest(),
                 hashlib.sha3_256(open(batch_image_paths[2], 'rb').read()).hexdigest(),
                 hashlib.sha3_256(open(batch_image_paths[3], 'rb').read()).hexdigest(),
                 hashlib.sha3_256(open(batch_image_paths[4], 'rb').read()).hexdigest()]
    print(f"Original SHA3-256 Hash: {sha3_hash[0]}")

    start1 = time.time()
    blur_dist = Hamming_distance(phashes[0], phashes[1])
    end1 = time.time()
    print('blur_dist is '+'%d' % blur_dist + ', similarity is ' +'%f' % (1 - blur_dist * 1.0 / 64) + ', time is ' +'%f' % (end1-start1) + ', phash is ' + phashes[1]+ ', sha3 is ' + sha3_hash[1])

    start2 = time.time()
    text_dist = Hamming_distance(phashes[0], phashes[2])
    end2 = time.time()
    print('text_dist is '+'%d' % text_dist + ', similarity is ' +'%f' % (1 - text_dist * 1.0 / 64) + ', time is ' +'%f' % (end2-start2) + ', phash is ' + phashes[2]+ ', sha3 is ' + sha3_hash[2])

    start3 = time.time()
    crop_dist = Hamming_distance(phashes[0], phashes[3])
    end3 = time.time()
    print('crop_dist is '+'%d' % crop_dist + ', similarity is ' +'%f' % (1 - crop_dist * 1.0 / 64) + ', time is ' +'%f' % (end3-start3) + ', phash is ' + phashes[3]+ ', sha3 is ' + sha3_hash[3])

    start4 = time.time()
    dot_dist = Hamming_distance(phashes[0], phashes[4])
    end4 = time.time()
    print('dot_dist is '+'%d' % dot_dist + ', similarity is ' +'%f' % (1 - dot_dist * 1.0 / 64) + ', time is ' +'%f' % (end4-start4) + ', phash is ' + phashes[4]+ ', sha3 is ' + sha3_hash[4])
    # Uncomment the following line to see the perceptual hashes for the batch of images
    # print(f"Perceptual Hashes (batch): {phashes}")