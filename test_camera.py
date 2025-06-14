import cv2
import time
import os

# Your exact Hikvision camera information
IP_ADDRESS = "172.16.101.193"
USERNAME = "admin"
PASSWORD = "SMART123"

print("------------------------------------------")
print("Hikvision CCTV Connection Test")
print("------------------------------------------")

# Use the exact URLs that work with VLC
vlc_url_1 = f"rtsp://{IP_ADDRESS}/0"
vlc_url_2 = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/0"

# Common Hikvision-specific URLs to try if VLC URLs fail
hikvision_urls = [
    # Main stream options
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/Streaming/Channels/101",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/h264/ch1/main/av_stream",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/h264/ch01/main/av_stream",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/Streaming/Channels/1",
    
    # Sub stream options
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/Streaming/Channels/102",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/h264/ch1/sub/av_stream",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/h264/ch01/sub/av_stream",
    f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}/Streaming/Channels/2",
]

# Combine VLC working URLs with Hikvision-specific URLs
urls_to_try = [vlc_url_1, vlc_url_2] + hikvision_urls

def test_connection(url):
    print(f"\nTrying to connect to: {url}")
    print("Please wait...")
    
    # Try to connect to the camera
    try:
        # Set FFMPEG as the backend (works better with RTSP)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # Give it a moment to connect
        time.sleep(2)
        
        # Check if connection worked
        if not cap.isOpened():
            print("❌ CONNECTION FAILED for this URL!")
            cap.release()
            return False, None
        
        # Try to read a frame to make sure streaming works
        ret, frame = cap.read()
        if not ret:
            print("❌ Connected but couldn't read video frames!")
            cap.release()
            return False, None
            
        print("✅ CONNECTION SUCCESSFUL!")
        return True, cap
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

# Try each URL until one works
success = False
working_url = None

try:
    for url in urls_to_try:
        success, cap = test_connection(url)
        if success:
            working_url = url
            break
    
    # If a URL worked
    if success:
        print(f"\n🎉 Successfully connected using: {working_url}")
        print("Now showing video feed for 10 seconds...")
        print("Press 'q' to quit early")
        
        # Save the working URL to a file
        with open("working_camera_url.txt", "w") as f:
            f.write(f"Working URL: {working_url}\n")
            f.write("Use this URL in your applications")
        
        # Display video for 10 seconds
        start_time = time.time()
        frames_displayed = 0
        
        while time.time() - start_time < 10:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if ret:
                frames_displayed += 1
                # Display the frame
                cv2.imshow("Hikvision Test", frame)
                
                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"⚠️ Connection lost after {frames_displayed} frames")
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"Test complete! Displayed {frames_displayed} frames.")
        print(f"Use this URL in your applications: {working_url}")
    
    # If none of the URLs worked
    else:
        print("\n❌ ALL CONNECTION ATTEMPTS FAILED!")
        print("\nTroubleshooting tips for Hikvision cameras:")
        print("1. Check your camera's web interface to verify RTSP is enabled")
        print("2. Verify username and password (try both lowercase and correct case)")
        print("3. Try accessing the camera through its web interface")
        print("4. Make sure your computer and camera are on the same network")
        print("5. Check if your camera requires a specific port (default is 554)")
        print("6. For Hikvision cameras, try adding ':554' to the URL")
        print("7. Check if VLC is still working with the URLs you provided")

except Exception as e:
    print(f"\n❌ Error during execution: {e}")

print("\nTest program completed.")