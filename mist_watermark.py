import os
import argparse
import subprocess
import cv2
import sys
from imwatermark import WatermarkEncoder, WatermarkDecoder

def add_watermark(input_path, output_path, message_text):
    """
    Stage 1: Embeds a robust invisible watermark (DWT+DCT) into the image.
    This happens BEFORE MIST touches the image.
    """
    print(f"[*] Stage 1: Embedding watermark '{message_text}'...")
    
    # 1. Read Image
    bgr = cv2.imread(input_path)
    if bgr is None:
        print(f"[!] Error: Could not read image at {input_path}")
        sys.exit(1)
        
    # 2. Configure Encoder 
    # 'dwtDct' (Discrete Wavelet + Cosine Transform) is robust against the 
    # high-frequency noise that MIST will add later.
    encoder = WatermarkEncoder()
    
    # We use 'bytes' to encode a string. 
    # Note: Keep the message short (e.g., "Copyright2025") to ensure robustness.
    encoder.set_watermark('bytes', message_text.encode('utf-8'))
    
    # 3. Embed
    bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')
    
    # 4. Save Intermediate Image
    cv2.imwrite(output_path, bgr_encoded)
    print(f"[+] Watermarked image saved to temporary file: {output_path}")

def run_mist(input_path, output_name, args):
    """
    Stage 2: Calls the existing mist_v3.py script using the watermarked image as input.
    """
    print(f"[*] Stage 2: Running MIST protection on {input_path}...")
    
    # Construct the command to call mist_v3.py
    # We pass through the relevant arguments from our wrapper
    cmd = [
        sys.executable, "mist_v3.py",
        "--input_image_path", input_path,
        "--output_name", output_name,
        "--epsilon", str(args.epsilon),
        "--steps", str(args.steps),
        "--mode", str(args.mode),
        "--rate", str(args.rate)
    ]
    
    if args.non_resize:
        cmd.append("--non_resize")
        
    # Execute MIST
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] MIST protection complete.")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running MIST: {e}")
        sys.exit(1)

def verify_watermark(image_path, msg_len):
    """
    Utility: Checks if the watermark survived the MIST process.
    """
    print(f"[*] Verifying watermark on final image: {image_path}...")
    bgr = cv2.imread(image_path)
    if bgr is None:
        return

    decoder = WatermarkDecoder('bytes', msg_len * 8)
    try:
        watermark = decoder.decode(bgr, 'dwtDct')
        decoded_msg = watermark.decode('utf-8')
        print(f"[?] Decoded Signature: '{decoded_msg}'")
        return decoded_msg
    except Exception as e:
        print(f"[-] Could not decode watermark. It may have been destroyed by MIST noise.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper for MIST to add Watermarking first.")
    
    # Wrapper Arguments
    parser.add_argument("-img", "--input_image_path", type=str, required=True, help="Path to original image")
    parser.add_argument("-wm", "--watermark_text", type=str, default="COPYRIGHT", help="Text to embed as watermark")
    parser.add_argument("--verify", action='store_true', help="Check watermark after processing")
    
    # MIST Pass-through Arguments (These match your mist_v3.py args)
    parser.add_argument("--output_name", type=str, default="protected_output", help="Name of output file")
    parser.add_argument("-e", "--epsilon", type=int, default=16, help="Strength of MIST attack")
    parser.add_argument("-n", "--steps", type=int, default=100, help="Number of MIST steps")
    parser.add_argument("-m", "--mode", type=int, default=2, help="MIST Mode (0:Semantic, 1:Texture, 2:Fused)")
    parser.add_argument("-r", "--rate", type=int, default=1, help="MIST Rate (exponent)")
    parser.add_argument("--non_resize", action='store_true', help="Do not resize image")

    args = parser.parse_args()

    # 1. define temporary filename for the watermarked version
    temp_wm_path = "temp_watermarked_input.png"
    
    try:
        # 2. Add Watermark
        add_watermark(args.input_image_path, temp_wm_path, args.watermark_text)
        
        # 3. Run MIST on the TEMPORARY watermarked file, not the original
        run_mist(temp_wm_path, args.output_name, args)
        
        # 4. Construct expected output path to verify
        # (MIST v3 usually saves to outputs/images/{output_name}_params.png)
        # We search for the file roughly because MIST appends params to the filename
        output_dir = os.path.join("outputs", "images")
        found_output = None
        for f in os.listdir(output_dir):
            if f.startswith(args.output_name) and f.endswith(".png"):
                found_output = os.path.join(output_dir, f)
                break
        
        # 5. Optional Verification
        if args.verify and found_output:
            verify_watermark(found_output, len(args.watermark_text))
            
    finally:
        # 6. Cleanup
        if os.path.exists(temp_wm_path):
            # os.remove(temp_wm_path)
            print("[*] Temporary files cleaned up.")