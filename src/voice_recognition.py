import cv2
import json
import keyboard
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pypylon import pylon
from queue import Queue

class LiveSpeechToText:
    def __init__(self, model_path="models/vosk-model-en-us-0.22", sample_rate=16000):
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        self.result_queue = Queue()
        self.final_results = []

    def _audio_callback(self, indata, frames, time, status):
        self.audio_buffer.append(indata.copy())

    def _toggle_recording(self):
        if not self.is_recording:
            # Start recording
            print("Listening...")
            self.audio_buffer = []
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                callback=self._audio_callback
            )
            self.stream.start()
        else:
            # Stop recording and process
            print("Processing...")
            self.stream.stop()
            self.stream.close()
            self.is_recording = False

            audio_data = np.concatenate(self.audio_buffer)
            audio_bytes = audio_data.tobytes()
            
            self.recognizer.Reset()
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                self.result_queue.put(result['text'])
            else:
                result = json.loads(self.recognizer.PartialResult())
                self.result_queue.put(result['partial'])

            # print(self.result_queue)

    def start(self):
        """
        Start listening for spacebar presses
        Returns: List of transcribed texts in order
        """
        keyboard.add_hotkey('space', self._toggle_recording)
        print("Press SPACE to start/stop recording. Press ESC to exit.")
        keyboard.wait('esc')
        
        # Collect all results from the queue
        while not self.result_queue.empty():
            self.final_results.append(self.result_queue.get())
            # print(self.final_results)
            
        return ' '.join(self.final_results)

class ImageStream:
    def __init__(self):
        self.webcam = None
        self.basler_cam = None

    def capture_webcam(self, output_path='webcam_image.jpg', camera_index=0):
        """Capture image from standard webcam using OpenCV"""
        try:
            self.webcam = cv2.VideoCapture(camera_index)
            if not self.webcam.isOpened():
                raise RuntimeError("Could not open webcam")

            # Allow camera to warm up
            for _ in range(5):
                self.webcam.read()

            ret, frame = self.webcam.read()
            if ret:
                cv2.imwrite(output_path, frame)
                return True
            return False
            
        finally:
            if self.webcam and self.webcam.isOpened():
                self.webcam.release()

    def capture_basler(self, output_path='basler_image.jpg'):
        """Capture image from Basler industrial camera using pypylon"""
        try:
            # Create camera instance
            self.basler_cam = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.basler_cam.Open()
            
            # Single shot mode
            self.basler_cam.StartGrabbingMax(1)
            grabResult = self.basler_cam.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                img = grabResult.Array
                cv2.imwrite(output_path, img)
                return True
            return False
            
        finally:
            if self.basler_cam and self.basler_cam.IsOpen():
                self.basler_cam.Close()

    def save_image_to_url():
        pass

# Example usage
if __name__ == '__main__':
    cam = ImageStream()
    
    # Capture from webcam
    if cam.capture_webcam(output_path='my_webcam.jpg'):
        print("Webcam capture successful")
    
    # Capture from Basler camera
    if cam.capture_basler(output_path='my_basler.jpg'):
        print("Basler capture successful")