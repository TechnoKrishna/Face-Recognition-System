import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk  # PIL is used to handle images in tkinter
import subprocess

# Define functions for each operation
def capture_faces():
    messagebox.showinfo("Capture Faces", "Starting face capture...")
    subprocess.run(["python", "capture_faces.py"])  # Assuming capture_faces.py is your script for face capture

def train_model():
    messagebox.showinfo("Train Model", "Training the model...")
    subprocess.run(["python", "train_model.py"])  # Assuming train_model.py is your script for training the model

def recognize_faces():
    messagebox.showinfo("Face Recognition", "Starting real-time face recognition...")
    subprocess.run(["python", "face_recognition.py"])  # Assuming recognize_faces.py is your script for face recognition

# Create the main window
root = tk.Tk()
root.title("Face Recognition System")

# Set window size and background color
root.geometry("500x400")
root.configure(bg='#2C3E50')

# Add a title label
title_label = tk.Label(root, text="Face Recognition System", bg='#2C3E50', fg='white', font=("Helvetica", 16, "bold"))
title_label.pack(pady=20)

# Load images for the buttons
capture_img = Image.open("icons/capture.png")
capture_img = capture_img.resize((50, 50), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
capture_icon = ImageTk.PhotoImage(capture_img)

train_img = Image.open("icons/train.png")
train_img = train_img.resize((50, 50), Image.LANCZOS)
train_icon = ImageTk.PhotoImage(train_img)

recognize_img = Image.open("icons/recognize.png")
recognize_img = recognize_img.resize((50, 50), Image.LANCZOS)
recognize_icon = ImageTk.PhotoImage(recognize_img)

# Add buttons for each operation, horizontally aligned
button_frame = tk.Frame(root, bg='#2C3E50')
button_frame.pack(pady=20)

capture_button = ttk.Button(button_frame, text="Capture Faces", command=capture_faces, image=capture_icon, compound="top", style="TButton")
capture_button.grid(row=0, column=0, padx=20)

train_button = ttk.Button(button_frame, text="Train Model", command=train_model, image=train_icon, compound="top", style="TButton")
train_button.grid(row=0, column=1, padx=20)

recognize_button = ttk.Button(button_frame, text="Face Recognition", command=recognize_faces, image=recognize_icon, compound="top", style="TButton")
recognize_button.grid(row=0, column=2, padx=20)

# Add a status bar
status_var = tk.StringVar()
status_var.set("Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg='#34495E', fg='white')
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Start the GUI event loop
root.mainloop()
