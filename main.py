import os
import math
import cv2
import cvzone
import kivy
import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, SlideTransition
from kivy.uix.scrollview import ScrollView
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
from scipy import ndimage
from ultralytics import YOLO


kivy.require('2.0.0')


KV = '''
ScreenManager:
    MainMenuScreen:
    RealTimeDetectionScreen:
    MediaDetectionScreen:
    StoreDetectedObjectsScreen:

<MainMenuScreen>:

    name: 'main_menu'
    BoxLayout:
        canvas.before:
            Color:
                rgba: 1, 0.647, 0, 1  # Orange color (Hex #FFA500)
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: 'vertical'
        MDToolbar:
            title: "Object Detection"
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [["menu", lambda x: app.callback(x), "Mai multe actiuni."]]
            right_action_items: [["close-circle", lambda x: app.exit_app(), "Exit app"]]  # Added exit button
            elevation: 10
        BoxLayout:
            orientation: 'vertical'
            spacing: dp(10)
            padding: dp(10)
            MDFillRoundFlatButton:
                text: "REAL-TIME DETECTION"
                font_size: 25
                pos_hint: {"center_x": 0.5, "center_y": 0.5}
                on_press: app.change_screen('real_time_detection')
            MDFillRoundFlatButton:
                text: "DETECT FROM MEDIA FILE"
                font_size: 25
                pos_hint: {"center_x": 0.5, "center_y": 0.5}
                on_press: app.change_screen('media_detection')

            MDFillRoundFlatButton:
                text: "STORE DETECTED OBJECTS"
                font_size: 25
                pos_hint: {"center_x": 0.5, "center_y": 0.5}
                on_press: app.change_screen('store_detected_objects')

<RealTimeDetectionScreen>:
    name: 'real_time_detection'
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: 0.765, 0.835, 0.667, 1  # Adjusted color values
            Rectangle:
                pos: self.pos
                size: self.size
        MDToolbar:
            title: "Real-Time Detection"
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [["arrow-left", lambda x: app.change_screen('main_menu')]]
        Image:
            id: real_time_image
            size_hint: (1, 1)
            allow_stretch: True
            keep_ratio: True
        BoxLayout:
            size_hint: (1, 0.1)
            pos_hint: {"center_x": 0.7, "center_y": 0.1}
            MDFillRoundFlatButton:
                id: start_button
                text: "START DETECTION"
                font_size: 25
                on_press: app.start_detection(self)
            MDFillRoundFlatButton:
                id: stop_button 
                text: "STOP DETECTION"
                font_size: 25
                on_press: app.stop_detection(self)


<MediaDetectionScreen>:
    name: 'media_detection'
    AnchorLayout:
        BoxLayout:
            id: box_layout  # Add this ID
            orientation: 'vertical'
            canvas.before:
                Color:
                    rgba: 1, 0.647, 0, 1  # Orange color (Hex #FFA500)
                Rectangle:
                    pos: self.pos
                    size: self.size
            MDToolbar:
                title: "Media Detection"
                md_bg_color: app.theme_cls.primary_color
                left_action_items: [["arrow-left", lambda x: app.change_screen('main_menu')]]
            BoxLayout:
                orientation: 'vertical'
                padding: dp(10)
                spacing: dp(10)
                MDFillRoundFlatButton:
                    text: "Upload Media File"
                    size_hint: None, None
                    size: 200, 50
                    pos_hint: {"center_x": 0.5, "center_y": 0.9}
                    on_press: app.file_chooser()
                FileChooserIconView:
                    id: filechooser
                    on_selection: app.select_file(self.selection)
                    size_hint_y: None
                    height: 0
                    pos_hint: {"center_x": 0.5, "center_y": 0.4}
                MDFillRoundFlatButton:
                    id: select_button
                    text: "Select or Show File"
                    size_hint: None, None
                    size: 200, 50
                    pos_hint: {"center_x": 0.5, "center_y": 0.4}
                    on_press: app.confirm_selection()

                BoxLayout:
                    size_hint: (1, 0.1)
                    pos_hint: {"center_x": 0.85, "center_y": 0.5}
                    MDFillRoundFlatButton:
                        text: "Detect Objects"
                        size_hint: None, None
                        size: 200, 50
                        pos_hint: {"center_x": 0.5, "center_y": 0.4}
                        on_press: app.process_media(self)
                    MDFillRoundFlatButton:
                        text: "Stop Detection"
                        size_hint: None, None
                        size: 200, 50
                        pos_hint: {"center_x": 0.5, "center_y": 0.4}
                        on_press: app.stop_media_detection()
                Image:
                    id: detected_media
                    size_hint: (1, 1)
                    allow_stretch: True
                    keep_ratio: True

<StoreDetectedObjectsScreen>:
    name: 'store_detected_objects'
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: 1, 0.647, 0, 1  # Blue color (Hex #4a9fe1)
            Rectangle:
                pos: self.pos
                size: self.size
        MDToolbar:
            title: "Store Detected Objects"
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [["arrow-left", lambda x: app.change_screen('main_menu')]]
        ScrollView:
            BoxLayout:
                id: table_layout
                orientation: "vertical"
                spacing: 10
                size_hint_y: None
                height: self.minimum_height



<ExitPopup>:
    title: "Exit Application"
    size_hint: None, None
    size: 400, 200
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        spacing: dp(10)
        padding: dp(10)
        MDLabel:
            text: "Are you sure you want to exit?"
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]
        BoxLayout:
            orientation: 'horizontal'
            spacing: dp(10)
            size_hint_y: None
            height: dp(50)
            pos_hint: {"center_x": 0.75, "center_y": 0.5}
            MDRaisedButton:
                text: "Yes"
                on_release: app.confirm_exit()
            MDRaisedButton:
                text: "No"
                on_release: root.dismiss()
'''
#Window.size = (400, 600)

class MainMenuScreen(Screen):
    pass


class ExitPopup(Popup):
    pass


class RealTimeDetectionScreen(Screen):
    pass


class MediaDetectionScreen(Screen):
    pass


class StoreDetectedObjectsScreen(Screen):
    pass


class ObjectDetectionApp(MDApp):

    def build(self):
        self.capture = None
        #self.model = YOLO("../Yolo-Weights/yolov8l.pt")
        self.model = YOLO("/home/dani/PycharmProjects/work/runs/detect/train36/weights/best.pt")
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        self.file_path = None  # To store the selected file path
        self.storeObjects = []
        Window.bind(on_dropfile=self.on_drop_file)
        self.detection_running = False
        self.video_capture = None
        self.current_media_type = None  # Initialize current_media_type
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Green'
        self.categories = None
        menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "Reset",
                "height": dp(56),
                "on_release": self.reset_detection,
            },
            {
                "viewclass": "OneLineListItem",
                "text": "More Information",
                "height": dp(56),
                "on_release": lambda: self.menu_callback("More Information"),
            }
        ]

        self.menu = MDDropdownMenu(
            items=menu_items,
            width_mult=4,
        )
        self.instructions = """Instructions:

        1. To reset the application, click on the "RESET" button.

        2. For real-time detection, click on the "REAL-TIME DETECTION" button.

        3. To detect objects from a media file, click on the "DETECT FROM MEDIA FILE" button.

        4. You can view and store detected objects by clicking on the "STORE DETECTED OBJECTS" button.
                """
        return Builder.load_string(KV)


    def create_objects_table(self, instance=None):
        # Get the table layout container
        categories = {
            "Fiinte": ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "Vehicule": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"],
            "Mobilier și Electrocasnice": ["chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
                                           "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                           "toaster", "sink", "refrigerator"],
            "       Instrumente și Obiecte Diverse": ["backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                                               "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                                               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
                                               "cup", "fork", "knife", "spoon", "bowl", "book", "clock", "vase",
                                               "scissors", "teddy bear", "hair drier", "toothbrush"],
            "Fructe": ["banana", "apple", "orange"],
            "Legume": ["broccoli", "carrot"],
            "Alimente Gătite": ["sandwich", "hot dog", "pizza", "donut", "cake"],
            "Obiecte din Mediu Urban": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
        }

        table_layout = self.root.get_screen('store_detected_objects').ids.table_layout
        table_layout.clear_widgets()

        # Create a GridLayout for the table with two columns
        grid = GridLayout(cols=2, spacing=5, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))

        # Add header to the grid
        grid.add_widget(
            Label(text="Category", size_hint_y=None, height=40, bold=True, halign='left', size_hint_x=None, width=110))
        grid.add_widget(
            Label(text="Objects", size_hint_y=None, height=40, bold=True, halign='left', size_hint_x=None, width=130))

        # Filter categories based on stored objects
        filtered_categories = {category: [obj for obj in objects if obj in self.storeObjects] for category, objects in
                               categories.items() if any(obj in self.storeObjects for obj in objects)}

        # Add categories and their objects to the grid
        for category, objects in filtered_categories.items():
            # Add category label
            grid.add_widget(
                Label(text=category, size_hint_y=None, height=30, halign='left', valign='middle', size_hint_x=None,
                      width=180))

            # Create a dynamic layout for objects
            objects_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10)
            objects_layout.bind(minimum_height=objects_layout.setter('height'))

            # Add objects to the layout, wrapping to new rows as needed
            row = BoxLayout(orientation='horizontal', size_hint_y=None, spacing=10)
            row.bind(minimum_height=row.setter('height'))
            max_width = self.root.width - 200  # Adjust based on your layout

            for obj in objects:
                obj_label = Label(text=obj, size_hint_y=None, height=30, size_hint_x=None, width=130)  # Set fixed width
                row.add_widget(obj_label)
                row_width = sum([child.width for child in row.children])

                if row_width > max_width:
                    objects_layout.add_widget(row)
                    row = BoxLayout(orientation='horizontal', size_hint_y=None, spacing=10)
                    row.bind(minimum_height=row.setter('height'))
                    obj_label = Label(text=obj, size_hint_y=None, height=30, size_hint_x=None,
                                      width=100)  # Set fixed width
                    row.add_widget(obj_label)

            # Add the final row if it has any children
            if row.children:
                objects_layout.add_widget(row)

            grid.add_widget(objects_layout)

        # Create a ScrollView to contain the grid
        scrollview = ScrollView(size_hint=(1, None), size=(self.root.width, self.root.height))
        scrollview.add_widget(grid)

        # Add the ScrollView to the table layout
        table_layout.add_widget(scrollview)

    # Call this function where necessary, for example, after detecting objects
    # create_objects_table(self)

    ### menu + realtime detection

    def change_screen(self, screen_name):
        self.root.transition = SlideTransition(direction='left')
        self.root.current = screen_name

    def start_detection(self, instance):
        if not self.detection_running:
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
                self.capture.set(3, 1280)
                self.capture.set(4, 720)
                Clock.schedule_interval(self.update, 1.0 / 30.0)
                self.detection_running = True
                Snackbar(text="Object detection started", duration=1).open()
            else:
                Snackbar(text="Camera is already in use", duration=1).open()
        else:
            Snackbar(text="Object detection is already running", duration=1).open()

    def stop_detection(self, instance):
        if self.detection_running:
            if self.capture is not None:
                Clock.unschedule(self.update)
                self.capture.release()
                self.capture = None
            self.detection_running = False
            self.root.get_screen('real_time_detection').ids.start_button.text = "RESUME DETECTION"
            Snackbar(text="Object detection is stopped", duration=1).open()
        else:
            Snackbar(text="Object detection is not running", duration=1).open()

    def update(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            print(f"Frame captured: {ret}")  # Debug print
            if ret:
                results = self.model(frame, stream=True)
                print(f"Results: {results}")  # Debug print
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(frame, (x1, y1, w, h))

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100))

                        # Classname
                        cls = int(box.cls[0])
                        if conf > 0.1:
                            cvzone.putTextRect(frame, f'{self.classNames[cls]} {conf}{"%"}', (max(0, x1), max(38, y1)),
                                               scale=1.5, thickness=2)
                        Objects = self.classNames[cls]
                        if Objects not in self.storeObjects:
                            self.storeObjects.append(Objects)

                # Display image from the camera
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tobytes()
                image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                print("Updating real-time image texture")  # Debug print
                self.root.get_screen('real_time_detection').ids.real_time_image.texture = image_texture

                self.create_objects_table()

    # photo detection + photo upload + photo drag

    def on_drop_file(self, window, file_path):
        self.file_path = file_path.decode("utf-8")
        if self.file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(self.file_path)
            if img is not None:
                # Convert the image to a Kivy-compatible format
                buf = cv2.flip(img, 0).tobytes()
                image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                # Display the selected image
                self.root.get_screen('media_detection').ids.detected_media.texture = image_texture
            Snackbar(text="File dropped: Ready to detect objects.", duration=2).open()
        elif self.file_path.lower().endswith(('.mp4', '.avi')):
            self.capture = cv2.VideoCapture(self.file_path)
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    # Display the first frame of the video
                    buf = cv2.flip(frame, 0).tobytes()
                    video_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.root.get_screen('media_detection').ids.detected_media.texture = video_texture
                    self.capture.release()  # Release the capture to avoid holding the video file
                    Snackbar(text="File dropped: Ready to detect objects.", duration=2).open()
                else:
                    print("Failed to read video frame")
                    Snackbar(text="Failed to read video frame", duration=2).open()
            else:
                print("Failed to open video")
                Snackbar(text="Failed to open video", duration=2).open()
        else:
            print("Unsupported file format")
            self.file_path = None
            Snackbar(text="Unsupported file format", duration=2).open()
        # Clear media from interface
        media_screen = self.root.get_screen('media_detection')
        media_screen.ids.detected_media.texture = None

    def file_chooser(self):
        filechooser = self.root.get_screen('media_detection').ids.filechooser
        select_button = self.root.get_screen('media_detection').ids.select_button
        if filechooser.height == 0:
            filechooser.height = 100  # Adjust the height as needed
            select_button.opacity = 1
        else:
            filechooser.height = 0
            select_button.opacity = 0

    def select_file(self, selection):
        if selection:
            self.file_path = selection[0]
            print(f"File selected: {self.file_path}")

    def confirm_selection(self):
        if self.file_path:
            print(f"File confirmed: {self.file_path}")
            Snackbar(text=f"File selected: {self.file_path}").open()
        else:
            Snackbar(text="No file selected").open()

    def process_media(self, instance):
        if self.file_path:
            if self.file_path.lower().endswith(('.mp4', '.avi')):
                self.detect_from_media_file('video', self.file_path)
            elif self.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                self.detect_from_media_file('image', self.file_path)
            else:
                print("Unsupported file type selected.")
                Snackbar(text="Unsupported file type selected.", duration=2).open()
        else:
            Snackbar(text="No file selected").open()

    def detect_from_media_file(self, media_type, media_path):
        self.file_path = media_path
        self.storeObjects = []  # Reset stored objects for each new file
        screen = self.root.get_screen('media_detection')

        if media_type == 'image':
            img = cv2.imread(self.file_path)
            if img is not None:
                # Convert the image to a Kivy-compatible format
                #img = self.enhance_image(img)
                buf = cv2.flip(img, 0).tobytes()
                image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                # Display the selected image
                screen.ids.detected_media.texture = image_texture
                print("Media displayed.")

                # Run your object detection algorithm on the media file
                results = self.model(img, stream=False)

                detected_objects = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(img, (x1, y1, w, h))

                        conf = math.ceil((box.conf[0] * 100))
                        cls = int(box.cls[0])
                        if conf > 0.1:
                            cvzone.putTextRect(img, f'{self.classNames[cls]} {conf}{"%"}', (max(0, x1), max(38, y1)),
                                               scale=1.5, thickness=2)
                            detected_objects.append(self.classNames[cls])
                            self.storeObjects.append(self.classNames[cls])  # Store detected objects

                # Convert the media file to a Kivy-compatible format
                buf = cv2.flip(img, 0).tobytes()
                image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                # Display the detected media file
                screen.ids.detected_media.texture = image_texture
                print("Detection complete and photo displayed.")

                # Display the detected objects
                detected_objects_str = ", ".join(detected_objects)
                Snackbar(text=f"Detected objects: {detected_objects_str}").open()
                self.create_objects_table()  # Update table with detected objects
            else:
                print("Failed to read the photo")
                Snackbar(text="Failed to read the photo", duration=1).open()

        elif media_type == 'video':
            self.capture = cv2.VideoCapture(self.file_path)
            if self.capture.isOpened():
                # Get video properties
                self.fps = self.capture.get(cv2.CAP_PROP_FPS)
                width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Create Kivy texture for video
                self.video_texture = Texture.create(size=(width, height), colorfmt='bgr')
                screen.ids.detected_media.texture = self.video_texture

                # Schedule video updates
                Clock.schedule_interval(self.update_video, 1.0 / self.fps)
                print("Video processing started.")
            else:
                print("Failed to open video.")
                Snackbar(text="Failed to open video.", duration=1).open()
        else:
            print("Unsupported file format")
            Snackbar(text="Unsupported file format", duration=2).open()

    def update_video(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            print("End of video or error reading frame.")
            Clock.unschedule(self.update_video)
            self.capture.release()
            return

        # Object detection on the frame
        self.detect_objects_in_frame(frame)

        # Display the processed frame
        buf = cv2.flip(frame, 0).tobytes()
        self.video_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def detect_objects_in_frame(self, frame):
        results = self.model(frame, stream=False)
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100))
                cls = int(box.cls[0])
                if conf > 0.1:
                    cvzone.putTextRect(frame, f'{self.classNames[cls]} {conf}{"%"}',
                                       (max(0, x1), max(38, y1)),
                                       scale=1.5, thickness=2)
                    detected_objects.append(self.classNames[cls])
                    if self.classNames[cls] not in self.storeObjects:
                        self.storeObjects.append(self.classNames[cls])

            detected_objects_str = ", ".join(detected_objects)
            Snackbar(text=f"Detected objects: {detected_objects_str}").open()
        self.create_objects_table()  # Update table with detected objects

    def stop_media_detection(self, instance=None):
        self.file_path = None
        # Stop video processing if running
        if hasattr(self, 'video_texture') and self.video_texture:
            Clock.unschedule(self.update_video)
            self.video_texture = None
            if self.capture:
                self.capture.release()  # Release the video capture object
                self.capture = None  # Reset the capture attribute
            Snackbar(text="Media detection stopped.").open()
        else:
            Snackbar(text="No file selected or media already stopped.").open()

    def reset_detection(self, instance=None, *args):
        # Clear stored objects and file path
        self.menu.dismiss()  # Close the menu
        self.storeObjects = []
        self.file_path = None

        # Clear media from interface
        media_screen = self.root.get_screen('media_detection')
        media_screen.ids.detected_media.texture = None
        self.root.get_screen('real_time_detection').ids.real_time_image.texture = None
        self.root.get_screen('real_time_detection').ids.start_button.text = "START DETECTION"

        # Clear the table layout
        table_layout = self.root.get_screen('store_detected_objects').ids.table_layout
        table_layout.clear_widgets()

        # Call create_objects_table to update the table with the cleared objects
        self.create_objects_table()

        # Stop real-time detection if running
        if self.capture:
            Clock.unschedule(self.update)
            self.capture.release()
            self.capture = None

        # Stop video processing if running
        if hasattr(self, 'video_texture') and self.video_texture:
            Clock.unschedule(self.update_video)
            self.video_texture = None

        Snackbar(text="Resetting detection...").open()

    def callback(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self, text_item):
        self.menu.dismiss()
        if text_item == "More Information":
            self.show_instructions()
        else:
            Snackbar(text=text_item).open()

    def show_instructions(self):
        popup = Popup(title='Instructions',
                      content=Label(text=self.instructions, text_size=(400, None), valign='top'),
                      size_hint=(None, None), size=(450, 400))
        popup.open()

    def exit_app(self):
        # Show the exit confirmation popup
        exit_popup = ExitPopup()
        exit_popup.open()

    def confirm_exit(self):
        # Exit the app
        MDApp.get_running_app().stop()
        Window.close()


if __name__ == '__main__':
    ObjectDetectionApp().run()
