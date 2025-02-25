import logging
import sys

# Configure logging first, before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure Kivy
import os
os.environ['KIVY_NO_CONSOLELOG'] = '1'  # Disable Kivy's default logging
os.environ['KIVY_TEXT'] = 'sdl2'  # Force SDL2 text provider
os.environ['KIVY_CLOCK'] = 'default'  # Force default clock
os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'  # Force SDL2 backend
os.environ['KIVY_KEYBOARD'] = 'sdl2'  # Force SDL2 keyboard
os.environ['KIVY_IMAGE'] = 'sdl2'  # Force SDL2 image provider
os.environ['KIVY_WINDOW'] = 'sdl2'  # Force SDL2 window provider
os.environ['KIVY_AUDIO'] = 'sdl2'  # Force SDL2 audio

from kivy.config import Config
Config.set('kivy', 'window', 'sdl2')
Config.set('kivy', 'text', 'sdl2')
Config.set('kivy', 'keyboard', 'sdl2')
Config.set('kivy', 'image', 'sdl2')
Config.set('kivy', 'audio', 'sdl2')
Config.set('kivy', 'kivy_clock', 'default')
Config.set('graphics', 'multisamples', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from kivy.config import Config
Config.set('kivy', 'text', 'sdl2')  # Explicitly set text provider
Config.set('kivy', 'kivy_clock', 'default')  # Set clock to default before other imports

import os
os.environ['KIVY_CLOCK'] = 'default'  # Force default clock

# Remove or comment out the direct clock import
# from kivy.clock import Clock

# Import Clock through App instead
from kivy.app import App
from kivy.clock import Clock as KivyClock  # Rename to avoid conflicts

import os
os.environ['KIVY_CLOCK'] = 'default'
# from kivy.clock import Clock

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
import cv2
import numpy as np
import mediapipe as mp
from math import hypot
import time
import requests
from io import BytesIO
from urllib.parse import quote
from threading import Thread
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.config import Config
import os
from kivy.uix.textinput import TextInput  # Add this import
from kivy.utils import get_color_from_hex
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.graphics import RoundedRectangle  # Add this import
from kivy.properties import BooleanProperty
from kivy.utils import platform
import threading
from functools import partial

# Replace pygame imports with jnius
from kivy.utils import platform
if platform == 'android':
    from jnius import autoclass
    MediaPlayer = autoclass('android.media.MediaPlayer')
    File = autoclass('java.io.File')
    Environment = autoclass('android.os.Environment')
    Context = autoclass('android.content.Context')
    FileOutputStream = autoclass('java.io.FileOutputStream')

# Add the current directory to resource path
resource_add_path(os.path.dirname(os.path.abspath(__file__)))

# Load the KV file
Builder.load_file('style.kv')

# Update window configuration before creating MainApp class
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'orientation', 'landscape')
Config.set('graphics', 'resizable', False)
Config.set('kivy', 'keyboard_mode', 'system')
Config.write()  # Save the configuration

# Let the system set window size based on screen dimensions
if platform == 'android':
    from jnius import autoclass
    activity = autoclass('org.kivy.android.PythonActivity').mActivity
    activity.setRequestedOrientation(0)  # SCREEN_ORIENTATION_LANDSCAPE
else:
    Window.size = (1024, 600)
    Window.rotation = 0
    if Window.width < Window.height:
        Window.size = (Window.height, Window.width)

class CameraWidget(Image):
    def __init__(self, **kwargs):
        try:
            super(CameraWidget, self).__init__(**kwargs)
            self.capture = None  # Don't initialize camera yet
            self.camera_index = 0  # Add default camera index
            self.front_camera_index = 1 if platform == 'android' else 0
            self.back_camera_index = 0 if platform == 'android' else 1
            self.left_ratio = 0.45  # Add default ratios
            self.right_ratio = 0.65
            
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Constants from test.py
            self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
            self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            self.LEFT_IRIS = [469, 470, 471, 472]
            self.RIGHT_IRIS = [474, 475, 476, 477]
            self.BLINK_THRESHOLD = 5.7
            
            # Initialize tracking variables
            self.frames = 0
            self.blinking_frames = 0
            self.frames_to_blink = 4  # Reduced from 7 to make it more responsive
            self.blink_cooldown = 0.3  # Shorter cooldown between blinks
            self.keyboard_selected = "left"
            self.last_keyboard_switch = time.time()
            self.keyboard_cooldown = 1.75
            self.last_blink_time = time.time()
            self.fps_time = time.time()
            self.frames_processed = 0
            self.current_fps = 0  # Add FPS counter variable
            
            Window.bind(size=self.on_window_resize)
            self.update_dimensions()
            self.register_event_type('on_gaze_direction')
            self.register_event_type('on_blink')
            self.current_gaze = "Center"
            self._update_thread = None
            self._stop_update = False
            self._last_frame = None  # Add frame buffer
            self.update_interval = 1.0/30.0  # 30 FPS
            self.texture = Texture.create(size=(640, 480), colorfmt='rgb')  # Changed from bgr to rgb
            self.texture.flip_vertical()  # Flip texture to correct orientation
        except Exception as e:
            logging.error(f"Error initializing CameraWidget: {e}", exc_info=True)
            raise

    def start_camera(self):
        if self.capture is not None:
            self.stop_camera()
            
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([
                    Permission.CAMERA,
                    Permission.WRITE_EXTERNAL_STORAGE,
                    Permission.READ_EXTERNAL_STORAGE
                ])
            except Exception as e:
                print(f"Error requesting Android permissions: {str(e)}")

        # Set camera properties before opening
        self.capture = cv2.VideoCapture(self.camera_index)
        if self.capture.isOpened():
            # Set camera resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Set camera FPS
            self.capture.set(cv2.CAP_PROP_FPS, 30)
        else:
            print(f"Failed to open camera {self.camera_index}")
            # Try the other index
            alt_index = self.front_camera_index if self.camera_index == self.back_camera_index else self.back_camera_index
            self.camera_index = alt_index
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                print("Failed to open any camera")
                return
        self._stop_update = False
        self._last_frame = None
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()

    def stop_camera(self):
        try:
            self._stop_update = True
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=1.0)
            if self.capture:
                self.capture.release()
                self.capture = None
            self._last_frame = None
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")

    def get_blinking_ratio(self, landmarks, img_h, img_w):
        try:
            # Convert landmarks to pixel coordinates
            landmark_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in landmarks
            ])
            
            # Get left eye points
            left_point = landmark_points[self.LEFT_EYE[0]]
            right_point = landmark_points[self.LEFT_EYE[3]]
            
            # Convert landmarks for center points calculation
            center_top = self.midpoint(
                landmark_points[self.LEFT_EYE[1]],
                landmark_points[self.LEFT_EYE[2]]
            )
            center_bottom = self.midpoint(
                landmark_points[self.LEFT_EYE[4]],
                landmark_points[self.LEFT_EYE[5]]
            )

            # Calculate distances
            hor_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
            ver_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

            ratio = hor_length / ver_length if ver_length > 0 else 0
            return ratio
        except Exception as e:
            print(f"Error calculating blink ratio: {str(e)}")
            return 0

    def midpoint(self, p1, p2):
        # Ensure p1 and p2 are numpy arrays or lists of coordinates
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    def iris_position(self, iris_center, right_point, left_point):
        try:
            if np.array_equal(iris_center, right_point) and np.array_equal(right_point, left_point):
                return "Unknown", 0
                    
            # Calculate distances exactly as in test.py
            ctr_dist = hypot(iris_center[0] - left_point[0], iris_center[1] - left_point[1])
            total_dist = hypot(right_point[0] - left_point[0], right_point[1] - left_point[1])
            
            ratio = ctr_dist / total_dist if total_dist > 0 else 0
            
            # Use instance variables for ratios
            if ratio <= self.left_ratio:
                return "Left", ratio
            elif ratio >= self.right_ratio:
                return "Right", ratio
            else:
                return "Center", ratio
        except Exception as e:
            print(f"Error calculating iris position: {str(e)}")
            return "Unknown", 0

    def _update_texture(self, dt):
        try:
            if self._last_frame is None:
                return
                
            frame = self._last_frame.copy()
            frame = cv2.resize(frame, (640, 480))
            
            # First do color conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS before any flips
            self.frames_processed += 1
            if time.time() - self.fps_time >= 1:
                self.current_fps = self.frames_processed
                self.frames_processed = 0
                self.fps_time = time.time()

            # Create text frame first
            text_frame = np.zeros_like(frame)
            
            # Draw FPS on text frame
            cv2.putText(text_frame, f"FPS: {self.current_fps}", 
                       (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)

            # Process face detection before flips
            results = self.face_mesh.process(frame)
            
            blinking_ratio = 0
            gaze_direction = "Unknown"
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                img_h, img_w = frame.shape[:2]
                
                # Calculate blinking ratio
                blinking_ratio = self.get_blinking_ratio(landmarks, img_h, img_w)
                
                if blinking_ratio > self.BLINK_THRESHOLD:
                    cv2.putText(text_frame, "BLINKING", (50, 150), 
                              cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=3)
                    self.blinking_frames += 1
                    self.frames -= 1
                    
                    if self.blinking_frames == self.frames_to_blink:
                        current_time = time.time()
                        if current_time - self.last_blink_time >= self.blink_cooldown:
                            self.last_blink_time = current_time
                            self.dispatch('on_blink', True)
                else:
                    self.blinking_frames = 0
                
                # Get iris landmarks and position
                iris_landmarks = np.array([
                    np.multiply([landmarks[idx].x, landmarks[idx].y], [img_w, img_h]).astype(int)
                    for idx in self.LEFT_IRIS
                ])
                
                iris_center = np.mean(iris_landmarks, axis=0).astype(int)
                right_point = np.multiply([landmarks[33].x, landmarks[33].y], [img_w, img_h]).astype(int)
                left_point = np.multiply([landmarks[133].x, landmarks[133].y], [img_w, img_h]).astype(int)
                
                gaze_direction, gaze_ratio = self.iris_position(iris_center, right_point, left_point)
                
                # Draw gaze direction on text frame
                cv2.putText(text_frame, f"Gaze: {gaze_direction}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw iris points
                for idx in self.LEFT_IRIS + self.RIGHT_IRIS:
                    x = int(landmarks[idx].x * img_w)
                    y = int(landmarks[idx].y * img_h)
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
                
                # Update gaze direction if changed
                if gaze_direction != self.current_gaze:
                    self.current_gaze = gaze_direction
                    self.dispatch('on_gaze_direction', gaze_direction)

            # Flip frames
            frame = cv2.flip(frame, 0)  # Vertical flip for display
            text_frame = cv2.flip(cv2.flip(text_frame, 1), 0)  # Horizontal then vertical flip for text
            
            # Combine frames
            final_frame = cv2.addWeighted(frame, 1, text_frame, 1, 0)
            
            # Update texture
            buf = final_frame.tobytes()
            texture = Texture.create(size=(final_frame.shape[1], final_frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.texture = texture
            
        except Exception as e:
            logging.error(f"Error updating texture: {e}", exc_info=False)
            return False

    def on_window_resize(self, instance, value):
        self.update_dimensions()

    def update_dimensions(self):
        window_width, window_height = Window.size
        target_width = window_width * 0.45
        target_height = window_height * 0.45
        
        self.size_hint = (None, None)
        self.size = (target_width, target_height)
        self.pos_hint = {'x': 0.02, 'top': 0.98}  # Add small margins

        # Draw border exactly at widget boundaries
        def on_pos(self, *args):
            self.canvas.after.clear()
            with self.canvas.after:
                Color(1, 0, 0, 1)
                Line(points=[self.x, self.y, self.x + self.width, self.y, 
                           self.x + self.width, self.y + self.height, 
                           self.x, self.y + self.height, self.x, self.y], width=2)
        
        self.bind(pos=on_pos, size=on_pos)

    def on_gaze_direction(self, *args):
        pass  # Default handler

    def on_blink(self, *args):
        pass  # Default handler

    def _update_loop(self):
        while not self._stop_update:
            try:
                if self.capture is None or not self.capture.isOpened():
                    break
                    
                ret, frame = self.capture.read()
                if ret:
                    # Flip the camera horizontally to create mirror effect
                    frame = cv2.flip(frame, 1)  # 1 for horizontal flip
                    self._last_frame = frame
                    # Schedule UI update on main thread
                    KivyClock.schedule_once(self._update_texture)
                    
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in camera update loop: {e}", exc_info=False)
                break

class KeyboardWidget(Image):
    def __init__(self, **kwargs):
        super(KeyboardWidget, self).__init__(**kwargs)
        self.keyboard = np.zeros((403, 500, 3), np.uint8)
        # Add both keyboard sets
        self.keys_set_1 = {
            0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
            5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
            10: "Z", 11: "X", 12: "C", 13: "V", 14: "<",
            15: "Sp", 16: "SN", 17: "RS"
        }
        self.keys_set_2 = {
            0: "Y", 1: "U", 2: "I", 3: "O", 4: "P",
            5: "H", 6: "J", 7: "K", 8: "L", 9: "_",
            10: "V", 11: "B", 12: "N", 13: "M", 14: "<",
            15: "Sp", 16: "SN", 17: "RS"
        }
        self.current_keyboard = "left"  # Track current keyboard
        self.letter_index = 0
        self.frames = 0
        self.frames_active_letter = 60  # Increased from 18 to slow down cursor
        KivyClock.schedule_interval(self.update, 1.0/30.0)  # Changed from Clock to KivyClock
        Window.bind(size=self.on_window_resize)
        self.update_dimensions()

    def on_window_resize(self, instance, value):
        self.update_dimensions()

    def update_dimensions(self):
        window_width, window_height = Window.size
        target_width = window_width * 0.45
        target_height = window_height * 0.45
        
        self.size_hint = (None, None)
        self.size = (target_width, target_height)
        self.pos_hint = {'right': 0.98, 'top': 0.98}  # Add small margins

        # Draw border exactly at widget boundaries
        def on_pos(self, *args):
            self.canvas.after.clear()
            with self.canvas.after:
                Color(1, 0, 0, 1)
                Line(points=[self.x, self.y, self.x + self.width, self.y,
                           self.x + self.width, self.y + self.height,
                           self.x, self.y + self.height, self.x, self.y], width=2)
        
        self.bind(pos=on_pos, size=on_pos)

    def draw_letters(self, letter_index, text, letter_light):
        key_positions = {
            0: (0, 0), 1: (100, 0), 2: (200, 0), 3: (300, 0), 4: (400, 0),
            5: (0, 100), 6: (100, 100), 7: (200, 100), 8: (300, 100), 9: (400, 100),
            10: (0, 200), 11: (100, 200), 12: (200, 200), 13: (300, 200), 14: (400, 200),
            15: (0, 300), 16: (100, 300), 17: (200, 300)
        }
        
        x, y = key_positions[letter_index]
        width = 100
        height = 100
        th = 3
        
        if letter_light:
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
            cv2.putText(self.keyboard, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 3, (51, 51, 51), 3)
        else:
            cv2.rectangle(self.keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
            cv2.putText(self.keyboard, text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    def get_active_letter(self):
        keys_set = self.keys_set_1 if self.current_keyboard == "left" else self.keys_set_2
        return keys_set[self.letter_index] if self.letter_index < len(keys_set) else None

    def update(self, dt):
        self.keyboard[:] = (0, 0, 0)
        self.frames += 1
        
        # Update letter selection with same timing as test.py
        if self.frames == self.frames_active_letter:
            self.letter_index += 1
            self.frames = 0
        if self.letter_index == 18:
            self.letter_index = 0
            
        # Select keyboard based on current_keyboard value
        keys_set = self.keys_set_1 if self.current_keyboard == "left" else self.keys_set_2
            
        # Draw keyboard
        for i in range(18):
            self.draw_letters(i, keys_set[i], i == self.letter_index)
        
        # Update active letter
        self.active_letter = self.get_active_letter()

        # Convert to texture
        rgb_keyboard = cv2.cvtColor(self.keyboard, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(rgb_keyboard, 0).tobytes()  # Changed from tostring()
        texture = Texture.create(size=(self.keyboard.shape[1], self.keyboard.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = texture

class BoardWidget(Image):
    def __init__(self, **kwargs):
        super(BoardWidget, self).__init__(**kwargs)
        self.board = np.zeros((600, 1400, 3), np.uint8)  # Increased height from 300 to 600
        self.board[:] = 255  # White background
        self.text = ""
        KivyClock.schedule_interval(self.update, 1.0/30.0)  # Changed from Clock to KivyClock
        Window.bind(size=self.on_window_resize)
        self.update_dimensions()

    def on_window_resize(self, instance, value):
        self.update_dimensions()

    def update_dimensions(self):
        window_width, window_height = Window.size
        target_width = window_width * 0.96
        target_height = window_height * 0.4
        
        self.size_hint = (None, None)
        self.size = (target_width, target_height)
        self.pos_hint = {'center_x': 0.5, 'y': 0.02}  # Center horizontally, margin from bottom

        # Draw red border with increased thickness
        def on_pos(self, *args):
            self.canvas.after.clear()
            with self.canvas.after:
                Color(1, 0, 0, 1)  # Pure red color
                Line(points=[0, 0, self.width, 0,
                           self.width, self.height,
                           0, self.height, 0, 0], 
                     width=4)  # Increased border width
        
        self.bind(pos=on_pos, size=on_pos)

    # Also update the text rendering to be more visible in the larger space
    def update(self, dt):
        self.board[:] = 255  # Clear board with white
        # Add text to board
        cv2.putText(self.board, self.text, (20, 300),  # Changed Y position to center text
                    cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 0), 4)  # Larger text
        
        # Convert to texture
        rgb_board = cv2.cvtColor(self.board, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(rgb_board, 0).tobytes()  # Changed from tostring()
        texture = Texture.create(size=(self.board.shape[1], self.board.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.texture = texture

    def update_text(self, new_text):
        self.text = new_text

class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)

    def go_to_main(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'main'

    def go_to_settings(self, *args):
        self.manager.transition.direction = 'left'
        self.manager.current = 'settings'

class SettingsScreen(Screen):
    is_front_camera = BooleanProperty(True)  # Declare as a property

    def __init__(self, app_instance, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)
        self.app = app_instance
        self.load_settings()

    def toggle_camera(self):
        self.is_front_camera = not self.is_front_camera  # Property will automatically update UI
        new_camera_index = 0 if self.is_front_camera else 1
        
        if new_camera_index != self.app.camera.camera_index:
            self.app.camera.camera_index = new_camera_index
            if self.app.camera.capture is not None:
                self.app.camera.start_camera()

    # Remove camera index validation from apply_settings since we're using toggle now
    def apply_settings(self, *args):
        try:
            blink = float(self.ids.blink_input.text)
            speed = int(self.ids.speed_input.text)
            left_ratio = float(self.ids.left_ratio_input.text)
            right_ratio = float(self.ids.right_ratio_input.text)
            
            # Validate all inputs
            if not (3.0 <= blink <= 8.0):
                self.show_error("Blink threshold must be between 3.0 and 8.0")
                return
            if not (0 <= speed <= 100):
                self.show_error("Cursor speed must be between 0 and 100")
                return
            if not (0.0 <= left_ratio <= 1.0):
                self.show_error("Left ratio must be between 0.0 and 1.0")
                return
            if not (0.0 <= right_ratio <= 1.0):
                self.show_error("Right ratio must be between 0.0 and 1.0")
                return
            if left_ratio >= right_ratio:
                self.show_error("Left ratio must be less than right ratio")
                return
                
            # Apply settings
            self.app.camera.BLINK_THRESHOLD = blink
            self.app.keyboard.frames_active_letter = speed
            self.app.camera.left_ratio = left_ratio
            self.app.camera.right_ratio = right_ratio
            
            # Save settings
            settings = {
                'blink': blink,
                'speed': speed,
                'left_ratio': left_ratio,
                'right_ratio': right_ratio
            }
            self.save_settings(settings)
                
        except ValueError:
            self.show_error("Please enter valid numbers")

    def show_error(self, message):
        error_label = Label(
            text=message,
            color=(1, 0, 0, 1),
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'y': 0.1}
        )
        self.add_widget(error_label)
        KivyClock.schedule_once(lambda dt: self.remove_widget(error_label), 2)  # Changed from Clock to KivyClock

    def go_back(self, *args):
        self.manager.transition.direction = 'right'
        self.manager.current = 'welcome'

    def load_settings(self):
        try:
            with open('settings.txt', 'r') as f:
                settings = eval(f.read())
                # Update UI
                self.ids.blink_input.text = str(settings.get('blink', 5.7))
                self.ids.speed_input.text = str(settings.get('speed', 60))
                self.ids.left_ratio_input.text = str(settings.get('left_ratio', 0.45))
                self.ids.right_ratio_input.text = str(settings.get('right_ratio', 0.65))
                # Apply settings immediately
                self.apply_settings()
        except:
            pass

    def save_settings(self, settings):
        try:
            with open('settings.txt', 'w') as f:
                f.write(str(settings))
        except Exception as e:
            print(f"Error saving settings: {str(e)}")

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)  # Add this line to attach layout to screen

    def on_enter(self):
        # Start camera when entering main screen
        if hasattr(self, 'app'):
            self.app.camera.start_camera()

    def on_leave(self):
        # Stop camera when leaving main screen
        if hasattr(self, 'app'):
            self.app.camera.stop_camera()

class MainApp(App):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MainApp, cls).__new__(cls)
            cls._instance.built = False  # Add this line
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_init'):
            super(MainApp, self).__init__()
            self.telegram_token = "7766310777:AAFwA2I9vlLA4olu_274lVuk4vkq74--fB4"
            self.telegram_chat_id = "1396515021"
            
            # Initialize audio system
            try:
                if platform == 'android':
                    # Get Android resources and assets
                    PythonActivity = autoclass('org.kivy.android.PythonActivity')
                    activity = PythonActivity.mActivity
                    Assets = autoclass('android.content.res.AssetManager')
                    assets = activity.getAssets()
                    
                    # Create MediaPlayer instances
                    self.sound = MediaPlayer()
                    self.left_sound = MediaPlayer()
                    self.right_sound = MediaPlayer()
                    
                    # Load sounds from assets
                    sound_files = {
                        'sound.wav': self.sound,
                        'left.wav': self.left_sound,
                        'right.wav': self.right_sound
                    }
                    
                    for filename, player in sound_files.items():
                        try:
                            asset_fd = assets.openFd(f'sounds/{filename}')
                            player.setDataSource(
                                asset_fd.getFileDescriptor(),
                                asset_fd.getStartOffset(),
                                asset_fd.getLength()
                            )
                            player.prepare()
                            print(f"Successfully loaded {filename}")
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                            player = None
                else:
                    self.sound = None
                    self.left_sound = None
                    self.right_sound = None
                    
            except Exception as e:
                print(f"Error initializing audio system: {str(e)}")
                self.sound = None
                self.left_sound = None
                self.right_sound = None
            
            self._init = True

    def play_sound(self, player):
        """Safe way to play Android MediaPlayer sound"""
        if platform == 'android' and player:
            try:
                if player.isPlaying():
                    player.stop()
                player.seekTo(0)
                player.start()
            except Exception as e:
                print(f"Error playing sound: {str(e)}")

    def speak_text(self, text):
        def speak():
            try:
                if platform == 'android':
                    # Use Android native TTS
                    try:
                        tts = autoclass('android.speech.tts.TextToSpeech')
                        PythonActivity = autoclass('org.kivy.android.PythonActivity')
                        context = PythonActivity.mActivity.getApplicationContext()
                        QUEUE_FLUSH = 0  # Constant from Android TTS
                        
                        def on_init(status):
                            if status == tts.SUCCESS:
                                result = tts_player.setLanguage(Locale.US)
                                if result in [tts.LANG_MISSING_DATA, tts.LANG_NOT_SUPPORTED]:
                                    print("Language not supported")
                                else:
                                    tts_player.speak(text, QUEUE_FLUSH, None)
                        
                        Locale = autoclass('java.util.Locale')
                        tts_player = tts(context, on_init)
                    except Exception as e:
                        print(f"Android TTS failed, falling back to web TTS: {str(e)}")
                        self._web_tts(text)
                else:
                    self._web_tts(text)
            except Exception as e:
                print(f"Error with text-to-speech: {str(e)}")
        
        Thread(target=speak).start()

    def _web_tts(self, text):
        try:
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q={quote(text)}"
            response = requests.get(url)
            if response.status_code == 200:
                if platform == 'android':
                    # Play audio using MediaPlayer on Android
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        f.write(response.content)
                        f.flush()
                        try:
                            player = MediaPlayer()
                            player.setDataSource(f.name)
                            player.prepare()
                            player.start()
                            # Clean up after playback
                            def on_completion():
                                player.release()
                                os.unlink(f.name)
                            player.setOnCompletionListener(on_completion)
                        except Exception as e:
                            print(f"Error playing TTS audio: {str(e)}")
                else:
                    print("Web TTS is only supported on Android")
            else:
                print(f"TTS request failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Web TTS error: {str(e)}")

    def build(self):
        # Ensure landscape orientation at startup
        if platform == 'android':
            # Android-specific orientation lock
            from jnius import autoclass
            activity = autoclass('org.kivy.android.PythonActivity').mActivity
            activity.setRequestedOrientation(0)  # SCREEN_ORIENTATION_LANDSCAPE
        elif Window.width < Window.height:
            # Desktop landscape correction
            Window.size = (Window.height, Window.width)

        Window.clearcolor = get_color_from_hex('#FAFAFA')
        
        # Create screen manager
        self.sm = ScreenManager()
        
        # Create screens
        welcome_screen = WelcomeScreen(name='welcome')
        settings_screen = SettingsScreen(self, name='settings')
        main_screen = MainScreen(name='main')
        main_screen.app = self  # Add reference to app
        
        # Add widgets to main screen
        self.camera = CameraWidget(
            size_hint=(0.25, 0.33),
            pos_hint={'x': 0, 'top': 1}
        )
        self.keyboard = KeyboardWidget()
        self.board = BoardWidget()
        
        main_screen.layout.add_widget(self.camera)
        main_screen.layout.add_widget(self.keyboard)
        main_screen.layout.add_widget(self.board)
        
        # Bind events
        self.camera.bind(on_gaze_direction=self.on_gaze_direction)
        self.camera.bind(on_blink=self.on_blink)
        
        # Add screens to manager
        self.sm.add_widget(welcome_screen)
        self.sm.add_widget(settings_screen)
        self.sm.add_widget(main_screen)
        
        settings_screen.load_settings()
        
        return self.sm

    def on_gaze_direction(self, instance, value):
        current_time = time.time()
        if current_time - self.camera.last_keyboard_switch >= self.camera.keyboard_cooldown:
            if (value == "Left" and self.keyboard.current_keyboard != "left") or \
               (value == "Right" and self.keyboard.current_keyboard != "right"):
                if value == "Left":
                    self.keyboard.current_keyboard = "left"
                    self.play_sound(self.left_sound)
                elif value == "Right":
                    self.keyboard.current_keyboard = "right"
                    self.play_sound(self.right_sound)
                self.camera.last_keyboard_switch = current_time

    def reset_app(self):
        # Reset board text
        self.board.text = ""
        self.board.update_text("")
        
        # Reset keyboard
        self.keyboard.letter_index = 0
        self.keyboard.frames = 0
        self.keyboard.current_keyboard = "left"
        
        # Reset camera tracking
        self.camera.blinking_frames = 0
        self.camera.frames = 0
        self.camera.current_gaze = "Center"
        self.camera.last_keyboard_switch = time.time()
        self.camera.last_blink_time = time.time()

    def on_blink(self, instance, value):
        active_letter = self.keyboard.active_letter
        if active_letter:
            self.play_sound(self.sound)
            
            if active_letter == "<":
                self.board.text = self.board.text[:-1]
            elif active_letter == "Sp":
                if self.board.text.strip():
                    self.speak_text(self.board.text)
            elif active_letter == "SN":
                try:
                    requests.post(
                        f'https://api.telegram.org/bot{self.telegram_token}/sendMessage',
                        params={'chat_id': self.telegram_chat_id, 'text': self.board.text}
                    )
                except Exception as e:
                    print(f"Error sending to Telegram: {str(e)}")
            elif active_letter == "RS":
                # Reset the application state
                self.reset_app()
            else:
                self.board.text += active_letter
            
            self.board.update_text(self.board.text)

    def on_stop(self):
        # Proper cleanup
        try:
            if platform == 'android':
                for player in [self.sound, self.left_sound, self.right_sound]:
                    if player:
                        player.release()
        except:
            pass
        cv2.destroyAllWindows()

# Remove pygame cleanup from main
if __name__ == '__main__':
    try:
        app = MainApp()
        app.run()
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()



