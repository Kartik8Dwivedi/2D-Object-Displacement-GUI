import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    return img1, img2

def find_paper_corners_multiple(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_corners = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            paper_corners.append(approx)

    return paper_corners

def compute_transformation_matrix(pts1, pts2):
    if len(pts1) == 4 and len(pts2) == 4:
        pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 1, 2)
        pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        
        if H is not None:
            H = np.round(H, 2)
        
        return H
    else:
        return None

def draw_paper_outline(img, points, color=(0, 0, 255)):
    if points is None or len(points) != 4:
        return img
    img_with_paper = img.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img_with_paper, [pts], isClosed=True, color=color, thickness=3)
    return img_with_paper

def get_transformation_matrices(img1, img2):
    corners1_list = find_paper_corners_multiple(img1)
    corners2_list = find_paper_corners_multiple(img2)

    if not corners1_list or not corners2_list:
        return "Failed to detect papers in one or both images."

    def compute_centroid(corners):
        return np.mean(corners.reshape(4, 2), axis=0)

    centroids1 = [compute_centroid(corners) for corners in corners1_list]
    centroids2 = [compute_centroid(corners) for corners in corners2_list]

    matched_pairs = []
    for i, c1 in enumerate(centroids1):
        distances = [np.linalg.norm(c1 - c2) for c2 in centroids2]
        min_dist_idx = np.argmin(distances)
        matched_pairs.append((corners1_list[i], corners2_list[min_dist_idx]))

    results = []
    for i, (pts1, pts2) in enumerate(matched_pairs):
        H = compute_transformation_matrix(pts1, pts2)
        if H is not None:
            results.append((i + 1, H))
        else:
            results.append((i + 1, "Failed to compute transformation matrix"))

    return results

def browse_image(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def process_images():
    image_path1 = img1_entry.get()
    image_path2 = img2_entry.get()

    if not image_path1 or not image_path2:
        messagebox.showerror("Error", "Please select both images.")
        return

    try:
        img1, img2 = load_images(image_path1, image_path2)
        results = get_transformation_matrices(img1, img2)

        output_text.delete(1.0, tk.END)
        for i, result in results:
            if isinstance(result, str):
                output_text.insert(tk.END, f"Paper {i}: {result}\n")
            else:
                output_text.insert(tk.END, f"Paper {i} Transformation Matrix:\n")
                
                zero_threshold = 1e-6

                for row in result:
                    formatted_row = ' '.join([f"{val:.2f}" if abs(val) > zero_threshold else "0.00" for val in row])
                    output_text.insert(tk.END, formatted_row + "\n")
                output_text.insert(tk.END, "\n")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


root = tk.Tk()
root.title("Transformation Matrix Detector")
root.geometry("600x400")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

img1_label = ttk.Label(frame, text="Image 1:")
img1_label.grid(row=0, column=0, sticky=tk.W)
img1_entry = ttk.Entry(frame, width=50)
img1_entry.grid(row=0, column=1, padx=5)
browse_img1_btn = ttk.Button(frame, text="Browse", command=lambda: browse_image(img1_entry))
browse_img1_btn.grid(row=0, column=2, padx=5)

img2_label = ttk.Label(frame, text="Image 2:")
img2_label.grid(row=1, column=0, sticky=tk.W)
img2_entry = ttk.Entry(frame, width=50)
img2_entry.grid(row=1, column=1, padx=5)
browse_img2_btn = ttk.Button(frame, text="Browse", command=lambda: browse_image(img2_entry))
browse_img2_btn.grid(row=1, column=2, padx=5)

process_btn = ttk.Button(frame, text="Get Transformation Matrix", command=process_images)
process_btn.grid(row=2, column=0, columnspan=3, pady=10)

output_text = tk.Text(frame, wrap="word", height=10, width=70)
output_text.grid(row=3, column=0, columnspan=3, pady=10)

root.mainloop()
