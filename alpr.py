import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import torch
import easyocr
import re
import csv
from datetime import datetime, time
import numpy as np
import os


class CarDetection:
    def __init__(self, capture_index, model_name, csv_filename='output.csv'):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cpu'
        self.ocr_reader = easyocr.Reader(['en'])
        self.csv_filename = csv_filename
        self.cap = None
        self.panel = None
        self.running = False
        self.background_image = None
        self.setup_gui()
        self.setup_csv()

    def setup_csv(self):
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Detected Text', 'Confidence', 'Timestamp'])

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def sharpen_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return sharpened

    def apply_clahe(self, gray_img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_img)

    def preprocess_for_ocr(self, cropped_img):
        sharpened = self.sharpen_image(cropped_img)
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe_img = self.apply_clahe(blurred)
        alpha = 2.0
        beta = -50
        enhanced_img = cv2.convertScaleAbs(clahe_img, alpha=alpha, beta=beta)
        binary = cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return processed_img

    def validate_indian_license_plate(self, text):
        patterns = [
            re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]\s?\d{4}$'),
            re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}$')
        ]
        return any(pattern.match(text) for pattern in patterns)

    def correct_common_errors(self, text):
        text = text.strip()
        text = re.sub(r'^[^\w]+', '', text)
        text = re.sub(r'\s+', '', text)
        corrected_text = []
        length = len(text)
        patterns = {
            10: ["L", "L", "D", "D", "L", "L", "D", "D", "D", "D"],
            9: ["L", "L", "D", "D", "L", "D", "D", "D", "D"]
        }
        expected_pattern = patterns.get(length, [])

        for i, char in enumerate(text):
            if i < len(expected_pattern):
                expected_type = expected_pattern[i]
                if expected_type == "L":
                    if char.lower() in '0123456789':
                        corrected_char = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G'}.get(char, char)
                    elif char == 'F' and i == 0:
                        corrected_char = 'P'
                    else:
                        corrected_char = char.upper()
                elif expected_type == "D":
                    if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        corrected_char = {'O': '0', 'I': '1', 'T': '1', 'D': '0', 'B': '8', 'Z': '2'}.get(char, char)
                    else:
                        corrected_char = char
            else:
                corrected_char = char

            corrected_text.append(corrected_char)

        return ''.join(corrected_text)

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2)

                cropped_img = frame[y1:y2, x1:x2]
                cropped_img = cv2.resize(cropped_img, (430, 210))
                preprocessed_img = self.preprocess_for_ocr(cropped_img)
                ocr_results = self.ocr_reader.readtext(preprocessed_img)

                for (bbox, text, prob) in ocr_results:
                    if len(text) > 4 and prob >= 0.5:
                        corrected_text = self.correct_common_errors(text)
                        if self.validate_indian_license_plate(corrected_text):
                            text_x, text_y = x1, y1 - 10
                            cv2.putText(frame, corrected_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 0, 255), 2)
                            self.license_plate_label.config(state=tk.NORMAL)
                            self.license_plate_label.delete(0, tk.END)
                            self.license_plate_label.insert(0, corrected_text)
                            self.license_plate_label.config(state='readonly')
                            timestamp = datetime.now()
                            self.date_label.config(state=tk.NORMAL)
                            self.date_label.delete(0, tk.END)
                            self.date_label.insert(0, f"{timestamp.date()}")
                            self.date_label.config(state='readonly')
                            self.time_label.config(state=tk.NORMAL)
                            self.time_label.delete(0, tk.END)
                            self.time_label.insert(0, f"{timestamp.time()}")
                            self.time_label.config(state='readonly')
                            self.write_to_csv(corrected_text)

        return frame

    def write_to_csv(self, text):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text, timestamp])

    def class_to_label(self, class_idx):
        return self.classes[int(class_idx)]

    def start_capture(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.capture_index)
        if not self.cap.isOpened():
            print("Error: Could not open video capture")
            return
        self.show_frame()

    def pause_capture(self):
        self.running = False

    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def show_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                if self.panel is None:
                    self.panel = tk.Label(self.video_frame, image=frame)
                    self.panel.image = frame
                    self.panel.pack(side="left", padx=10, pady=10)
                else:
                    self.panel.configure(image=frame)
                    self.panel.image = frame
            self.root.after(10, self.show_frame)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Car Detection System")
        self.root.geometry("1200x700")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.video_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.video_frame.grid(row=0, column=0, rowspan=2, columnspan=2, padx=10, pady=10, sticky='nsew')

        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky='ew')

        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start_capture, bg="green",
                                      fg="white", width=15, height=2, font=("Helvetica", 14))
        self.start_button.pack(side=tk.LEFT, padx=20, pady=10, expand=True)

        self.pause_button = tk.Button(self.control_frame, text="Pause", command=self.pause_capture, bg="orange",
                                      fg="white", width=15, height=2, font=("Helvetica", 14))
        self.pause_button.pack(side=tk.LEFT, padx=20, pady=10, expand=True)

        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop_capture, bg="red", fg="white",
                                     width=15, height=2, font=("Helvetica", 14))
        self.stop_button.pack(side=tk.LEFT, padx=20, pady=10, expand=True)

        self.info_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.info_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky='nsew')

        self.license_plate_label = self.create_label_entry(self.info_frame, "License Plate:", 0)
        self.date_label = self.create_label_entry(self.info_frame, "Date:", 1)
        self.time_label = self.create_label_entry(self.info_frame, "Time:", 2)

        self.query_button = tk.Button(self.info_frame, text="Query", command=self.open_query_window, bg="blue",
                                      fg="white", width=15, height=2, font=("Helvetica", 14))
        self.query_button.grid(row=6, column=0, padx=20, pady=10, sticky='ew')

        self.root.mainloop()

    def create_label_entry(self, parent, text, row):
        frame = tk.Frame(parent)
        frame.grid(row=row * 2, column=0, padx=10, pady=5, sticky='nsew')

        label = ttk.Label(frame, text=text, font=("Helvetica", 16))
        label.pack(anchor='center')

        entry = ttk.Entry(frame, state='readonly', font=("Helvetica", 14))
        entry.pack(anchor='center')

        return entry

    def open_query_window(self):
        query_window = tk.Toplevel(self.root)
        query_window.title("Query License Plates")
        query_window.geometry("400x700")

        date_label = tk.Label(query_window, text="Date (YYYY-MM-DD):", font=("Helvetica", 14))
        date_label.pack(pady=10)
        self.query_date_entry = tk.Entry(query_window, font=("Helvetica", 14))
        self.query_date_entry.pack(pady=10)

        start_time_label = tk.Label(query_window, text="Start Time (HH:MM:SS):", font=("Helvetica", 14))
        start_time_label.pack(pady=10)
        self.query_start_time_entry = tk.Entry(query_window, font=("Helvetica", 14))
        self.query_start_time_entry.pack(pady=10)

        end_time_label = tk.Label(query_window, text="End Time (HH:MM:SS):", font=("Helvetica", 14))
        end_time_label.pack(pady=10)
        self.query_end_time_entry = tk.Entry(query_window, font=("Helvetica", 14))
        self.query_end_time_entry.pack(pady=10)

        plate_label = tk.Label(query_window, text="License Plate:", font=("Helvetica", 14))
        plate_label.pack(pady=10)
        self.query_plate_entry = tk.Entry(query_window, font=("Helvetica", 14))
        self.query_plate_entry.pack(pady=10)

        submit_button = tk.Button(query_window, text="Submit", command=self.query_csv, font=("Helvetica", 14))
        submit_button.pack(pady=20)

        self.query_result_text = tk.Text(query_window, height=10, width=50, font=("Helvetica", 12))
        self.query_result_text.pack(pady=10)

    def query_csv(self):
        query_date = self.query_date_entry.get()
        query_start_time = self.query_start_time_entry.get()
        query_end_time = self.query_end_time_entry.get()
        query_plate = self.query_plate_entry.get()

        if query_plate:
            results = []
            with open(self.csv_filename, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    detected_text, timestamp_str = row
                    if detected_text == query_plate:
                        results.append(f"{detected_text} detected at {timestamp_str}")

            self.query_result_text.delete(1.0, tk.END)
            if results:
                self.query_result_text.insert(tk.END, "\n".join(results) + "\n")
            else:
                self.query_result_text.insert(tk.END, "No results found for the given license plate.\n")
        else:
            if not query_date or not query_start_time or not query_end_time:
                self.query_result_text.insert(tk.END, "Please fill in all fields.\n")
                return

            query_start_datetime = datetime.strptime(f"{query_date} {query_start_time}", '%Y-%m-%d %H:%M:%S')
            query_end_datetime = datetime.strptime(f"{query_date} {query_end_time}", '%Y-%m-%d %H:%M:%S')

            results = []
            with open(self.csv_filename, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    detected_text, timestamp_str = row
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    if query_start_datetime <= timestamp <= query_end_datetime:
                        results.append(detected_text)

            self.query_result_text.delete(1.0, tk.END)
            if results:
                self.query_result_text.insert(tk.END, "\n".join(results) + "\n")
            else:
                self.query_result_text.insert(tk.END, "No results found for the given time range.\n")


if __name__ == "__main__":
    CarDetection(capture_index='3.mov', model_name='best.pt')
