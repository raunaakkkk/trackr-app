import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import subprocess
import threading
import sys
import requests
import base64

# ==========================================
# AUTO-LIVE FEATURE DEPLOYMENT LAUNCHER
# ==========================================
def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    if os.environ.get("IS_STREAMLIT_RUNNING") == "1":
        return True
    if len(sys.argv) > 0 and 'streamlit' in sys.argv[0]:
        return True
    return False

if not is_running_in_streamlit():
    script_path = os.path.abspath(__file__)
    print("="*70)
    print("🚀 Starting AI Talent Scout & Secure Auto-Tunnel...")
    print("="*70)
    
    env = os.environ.copy()
    env["IS_STREAMLIT_RUNNING"] = "1"
    
    # 1. Start the Streamlit Application as a Subprocess child.
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", "8501", "--server.headless", "true"],
        env=env
    )
    
    print("\n⏳ Mapping Cloudflare / Localtunnel Route to external edge...")
    
    # 2. Launch LocalTunnel
    lt_proc = subprocess.Popen(
        "npx localtunnel --port 8501",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 3. Thread for parsing localtunnel output live
    def print_tunnel_output():
        for line in lt_proc.stdout:
            line = line.strip()
            if line:
                if "your url is:" in line.lower():
                    print("\n" + "🌟"*35)
                    print(f"\n   LIVE PUBLIC APP URL:  {line.split(': ', 1)[-1]}\n")
                    print("👉 Click the link above to test your AI Talent Scout app natively on your mobile phone / browser!")
                    print("🌟"*35 + "\n")
                else:
                    pass # Hide tunnel diagnostics for a cleaner terminal
                    
    lt_thread = threading.Thread(target=print_tunnel_output, daemon=True)
    lt_thread.start()
    
    # Wait until user closes the application in terminal
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down AI Talent Scout server safely...")
        streamlit_proc.terminate()
        lt_proc.terminate()
        
    # Prevent Python from executing the rest of the file bare-bones outside Streamlit context
    sys.exit(0)

# ==========================================
# SYSTEM DIAGNOSTICS
# ==========================================
def check_system_status():
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    
    st.sidebar.subheader("🌐 System Status")
    
    # Twilio Status
    if sid and token:
        st.sidebar.success("✅ TURN Server: Configured")
    else:
        st.sidebar.warning("⚠️ TURN Server: Not Found")
        st.sidebar.caption("Video might not work on mobile data without Twilio SID/Token environment variables.")

    # Camera/WebRTC status info
    st.sidebar.info("💡 Hint: If the camera doesn't start, ensure you are using HTTPS or localhost.")

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Talent Scout", layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏆 AI Talent Scout</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #888;'>Perform live AI-driven assessments directly in your browser.</h4>", unsafe_allow_html=True)
st.divider()

st.sidebar.title("🛠️ Athlete Config")
mode = st.sidebar.radio("Assessment Mode", ["Sit-up", "Vertical Jump"])
user_height_cm = st.sidebar.number_input("User Height (cm)", min_value=50, max_value=250, value=170, step=1)

# Status diagnostics
st.sidebar.divider()
check_system_status()
st.sidebar.divider()

# Display context instructions based on mode
if mode == "Sit-up":
    st.sidebar.info("📌 **Sit-up Instructions:**\n1. Place camera on the floor.\n2. Ensure your full side-profile (Shoulder, Hip, Knee) is visible.\n3. Perform full sit-ups!")
else:
    st.sidebar.info("📌 **Vertical Jump Instructions:**\n1. Place camera on the floor facing you.\n2. Stand fully visible from head to toe.\n3. Wait 1-2 seconds for your baseline height to calibrate.\n4. Jump as high as you can!")



# ==========================================
# TWILIO TURN SERVER CONFIGURATION
# ==========================================
@st.cache_data(ttl=3600)
def get_ice_servers():
    """
    Fetches ICE servers from Twilio's REST API.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    if account_sid and auth_token:
        try:
            auth = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json",
                headers={"Authorization": f"Basic {auth}"},
                timeout=5
            )
            if response.status_code == 201:
                return response.json().get("ice_servers")
        except Exception:
            pass

    return [{"urls": ["stun:stun.l.google.com:19302"]}]

rtc_configuration = RTCConfiguration({"iceServers": get_ice_servers()})

# ==========================================
# CV TARGET CALCULATION LOGIC
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle between three points a, b, c. 
    Point 'b' is the vertex.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# ==========================================
# WEBRTC VIDEO PROCESSOR CLASS
# ==========================================
class AssessmentVideoProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Live Configurations
        self.mode = "Sit-up"
        self.user_height_cm = 170
        
        # --- Internal States ---
        
        # Sit-up state
        self.situp_state = "up"
        self.reps = 0
        
        # Vertical Jump state
        self.baseline_y = None
        self.min_y = float('inf')
        self.jump_frames = 0
        self.max_jump_cm = 0.0

    def recv(self, frame):
        # Convert frame from WebRTC layout
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        
        # Mediapipe requires RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)
        img_rgb.flags.writeable = True
        
        if results.pose_landmarks:
            # Draw MediaPipe Skeleton
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # ---------------------
            # SIT-UP LOGIC
            # ---------------------
            if self.mode == "Sit-up":
                try:
                    # Extract 2D locations for Left Shoulder, Hip, and Knee
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    angle = calculate_angle(shoulder, hip, knee)

                    # State Machine logic matching prompt specifications
                    if angle > 150:
                        self.situp_state = "down"
                    if angle < 50 and self.situp_state == "down":
                        self.situp_state = "up"
                        self.reps += 1

                    # Overlay logic mapping to Oversized Native Metrics 
                    cv2.putText(img, f'REPS: {self.reps}', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 5, cv2.LINE_AA)
                    cv2.putText(img, f'Hip Angle: {int(angle)}', (45, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, f'State: {self.situp_state.upper()}', (45, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass

            # ---------------------
            # VERTICAL JUMP LOGIC
            # ---------------------
            elif self.mode == "Vertical Jump":
                try:
                    # Capture Mid-hip Y Coordinate
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    mid_hip_y = (left_hip.y + right_hip.y) / 2.0 * H

                    # Extrapolate full body pixel height from Eye to Heel for scaling
                    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
                    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
                    body_pixel_height = abs(left_heel.y - left_eye.y) * H

                    # Phase 1: Callibrate the first 30 frames to establish 'baseline_y' coordinate
                    if self.jump_frames < 30:
                        if self.baseline_y is None:
                            self.baseline_y = mid_hip_y
                        else:
                            # Moving Average
                            self.baseline_y = (self.baseline_y * self.jump_frames + mid_hip_y) / (self.jump_frames + 1)
                        self.jump_frames += 1
                        
                        # Show calibration text
                        cv2.putText(img, f'Calibrating...', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 165, 255), 5, cv2.LINE_AA)
                        cv2.putText(img, f'Stay still! {self.jump_frames}/30 frames', (45, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    # Phase 2: Track peak jump minimum Y value actively
                    else:
                        if mid_hip_y < self.min_y:
                            self.min_y = mid_hip_y
                            
                            # Y increases downwards, so displacement is baseline - min
                            pixel_displacement = self.baseline_y - self.min_y
                            if body_pixel_height > 0:
                                self.max_jump_cm = (pixel_displacement / body_pixel_height) * self.user_height_cm

                        # Display oversized metrics natively
                        cv2.putText(img, f'MAX JUMP: {self.max_jump_cm:.1f} cm', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, 'Ready! Jump Now!', (45, 160), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================================
# MAIN APP BODY
# ==========================================
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown(f"### {mode} Tracker Feed", unsafe_allow_html=True)
    
    ctx = webrtc_streamer(
        key="ai-talent-scout",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=AssessmentVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Inject current streamlit states into the active WebRTC video thread class
    if ctx.video_processor:
        ctx.video_processor.mode = mode
        ctx.video_processor.user_height_cm = user_height_cm
