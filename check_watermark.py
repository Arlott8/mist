import argparse
import cv2
import sys
from imwatermark import WatermarkDecoder

def calculate_similarity(s1, s2):
    """
    Calculates percentage of matching characters.
    """
    if len(s1) != len(s2):
        return 0.0
    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    return matches / len(s1)

def verify_watermark(image_path, expected_text, tolerance=0.8):
    print(f"[*] Analyzing image: {image_path}")
    
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[!] Error: Could not load image at {image_path}")
        return False

    # 1. Calculate Bits
    expected_bytes = expected_text.encode('utf-8')
    bit_length = len(expected_bytes) * 8
    
    # 2. Decode
    decoder = WatermarkDecoder('bytes', bit_length)

    try:
        # Try decoding
        watermark = decoder.decode(bgr, 'dwtDctSvd')
        decoded_text = watermark.decode('utf-8', errors='replace') # 'replace' avoids crashing on bad chars
        
        # 3. Fuzzy Compare
        similarity = calculate_similarity(expected_text, decoded_text)
        print(f"    Found Hidden Text: '{decoded_text}'")
        print(f"    Similarity Score:  {similarity:.0%}")
        
        if similarity >= tolerance:
            print(f"[+] SUCCESS: Watermark verified! (>{tolerance:.0%} match)")
            return True
        else:
            print(f"[-] FAILURE: Watermark too corrupted.")
            print(f"    (Expected: '{expected_text}' vs Found: '{decoded_text}')")
            return False

    except Exception as e:
        print(f"[-] Error during decoding: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument("text", help="Expected watermark text")
    args = parser.parse_args()
    
    verify_watermark(args.image, args.text)