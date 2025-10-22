# gui/app.py

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import time
import os
from pipeline.orchestrator import run_full_pipeline
from PIL import Image # <-- NEW: Import Pillow

class Redirector:
    """Redirects print statements to a tkinter widget."""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.after(0, self.update_text, text)

    def update_text(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Multi-Agent AutoML Pipeline 🤖")
        self.geometry("1100x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # --- Layout Configuration ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Navigation) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Project Steps", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.step_labels = []
        steps = [
            "1. Data Acquisition", "2. Problem Analysis", "3. Preprocessing",
            "4. Visualization", "5. Feature Engineering", "6. Data Staging",
            "7. AutoML Training", "8. Export Model"
        ]
        for i, step in enumerate(steps):
            label = ctk.CTkLabel(self.sidebar_frame, text=step, anchor="w", font=ctk.CTkFont(size=14))
            label.grid(row=i+1, column=0, padx=20, pady=10, sticky="ew")
            self.step_labels.append(label)

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=1) # Give weight to log frame
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.step1_title = ctk.CTkLabel(self.main_frame, text="Step 1: Data Acquisition", font=ctk.CTkFont(size=18, weight="bold"))
        self.step1_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10), sticky="w")
        
        # --- Tab View for Data Acquisition ---
        self.tab_view = ctk.CTkTabview(self.main_frame, height=200)
        self.tab_view.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="ew")

        # --- Tab 1: Upload File ---
        self.tab_view.add("Upload File")
        self.tab_view.tab("Upload File").grid_columnconfigure(1, weight=1)
        self.upload_button = ctk.CTkButton(self.tab_view.tab("Upload File"), text="Upload CSV File", command=self.upload_file)
        self.upload_button.grid(row=0, column=0, padx=20, pady=20)
        self.filepath_label = ctk.CTkLabel(self.tab_view.tab("Upload File"), text="No file selected.", text_color="gray")
        self.filepath_label.grid(row=0, column=1, padx=20, pady=20, sticky="w")

        # --- Tab 2: Search Online ---
        self.tab_view.add("Search Online")
        self.tab_view.tab("Search Online").grid_columnconfigure(1, weight=1)
        self.search_label = ctk.CTkLabel(self.tab_view.tab("Search Online"), text="Search Query:")
        self.search_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        self.search_entry = ctk.CTkEntry(self.tab_view.tab("Search Online"), placeholder_text="e.g., 'customer churn data' (Feature coming soon)", width=400)
        self.search_entry.grid(row=0, column=1, padx=20, pady=20, sticky="ew")
        
        # --- Tab 3: Generate Synthetic ---
        self.tab_view.add("Generate Synthetic")
        self.tab_view.tab("Generate Synthetic").grid_columnconfigure(1, weight=1)
        self.generate_label = ctk.CTkLabel(self.tab_view.tab("Generate Synthetic"), text="Data Description:")
        self.generate_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        self.generate_entry = ctk.CTkEntry(self.tab_view.tab("Generate Synthetic"), placeholder_text="e.g., '100 rows, 5 features, classification task about bank loans'", width=400)
        self.generate_entry.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

        # --- Problem Description (Common to all tabs) ---
        self.problem_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.problem_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=0, sticky="ew")
        self.problem_frame.grid_columnconfigure(1, weight=1)
        self.problem_label = ctk.CTkLabel(self.problem_frame, text="Describe your problem:")
        self.problem_label.grid(row=0, column=0, padx=0, pady=10, sticky="w")
        self.problem_entry = ctk.CTkEntry(self.problem_frame, placeholder_text="e.g., 'Predict if a customer will churn based on their account info'", height=40)
        self.problem_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        # --- NEW: Results Tab View (Log and Charts) ---
        self.results_tab_view = ctk.CTkTabview(self.main_frame, height=250)
        self.results_tab_view.grid(row=4, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        
        # Log Tab
        self.log_tab = self.results_tab_view.add("Progress Log")
        self.log_tab.grid_columnconfigure(0, weight=1)
        self.log_tab.grid_rowconfigure(0, weight=1)
        self.log_textbox = ctk.CTkTextbox(self.log_tab, state="disabled", wrap="word")
        self.log_textbox.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Charts Tab
        self.charts_tab = self.results_tab_view.add("Charts")
        self.charts_tab.grid_columnconfigure(0, weight=1)
        self.charts_tab.grid_rowconfigure(0, weight=1)
        self.charts_frame = ctk.CTkScrollableFrame(self.charts_tab, label_text="Generated Visualizations")
        self.charts_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # --- Start Button & Progress Bar ---
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.control_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=0, sticky="ew")
        self.control_frame.grid_columnconfigure(0, weight=1)

        self.start_button = ctk.CTkButton(self.control_frame, text="🚀 Start AutoML Pipeline", command=self.start_pipeline, height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.start_button.grid(row=0, column=0, padx=0, pady=10, sticky="ew")
        
        self.progress_bar = ctk.CTkProgressBar(self.control_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
        
        # --- Redirect print to log_textbox ---
        sys.stdout = Redirector(self.log_textbox)

    def upload_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.filepath_label.configure(text=filepath, text_color="white")
    
    def update_step_highlight(self, current_step_index):
        for i, label in enumerate(self.step_labels):
            if i == current_step_index:
                label.configure(font=ctk.CTkFont(size=14, weight="bold"), text_color="cyan")
            else:
                label.configure(font=ctk.CTkFont(size=14, weight="normal"), text_color="white")

    def pipeline_callback(self, message, progress):
        try:
            current_step_index = int(message.split(":")[0].split(" ")[1]) - 1
            self.after(0, self.update_gui, message, progress, current_step_index)
        except:
             self.after(0, self.update_gui, message, progress, -1)
        
    def update_gui(self, message, progress, current_step_index):
        print(f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.progress_bar.set(progress / 100)
        if current_step_index != -1:
            self.update_step_highlight(current_step_index)

    def start_pipeline(self):
        problem_desc = self.problem_entry.get()
        if not problem_desc:
            messagebox.showerror("Error", "Please describe your problem.")
            return

        active_tab = self.tab_view.get()
        acquisition_mode = ""
        acquisition_input = ""

        if active_tab == "Upload File":
            filepath = self.filepath_label.cget("text")
            if "No file selected" in filepath:
                messagebox.showerror("Error", "Please select a dataset to upload.")
                return
            acquisition_mode = "upload"
            acquisition_input = filepath
        
        elif active_tab == "Search Online":
            messagebox.showinfo("Info", "Search feature not yet implemented. Please upload a file or generate data.")
            return
            
        elif active_tab == "Generate Synthetic":
            description = self.generate_entry.get()
            if not description:
                messagebox.showerror("Error", "Please describe the synthetic data to generate.")
                return
            acquisition_mode = "generate"
            acquisition_input = description

        self.start_button.configure(state="disabled", text="Pipeline is Running...")
        
        # Clear log
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", tk.END)
        self.log_textbox.configure(state="disabled")
        
        # Clear previous charts
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
            
        # Switch to log tab
        self.results_tab_view.set("Progress Log")
        
        print("Starting the AutoML pipeline...\n")
        
        pipeline_state = {
            "acquisition_mode": acquisition_mode,
            "acquisition_input": acquisition_input,
            "problem_description": problem_desc,
            "callback": self.pipeline_callback,
            "results_dir": os.path.join(os.getcwd(), "results")
        }
        
        pipeline_thread = threading.Thread(
            target=self.run_pipeline_thread, 
            args=(pipeline_state,), 
            daemon=True
        )
        pipeline_thread.start()

    def run_pipeline_thread(self, pipeline_state):
        try:
            result_state = run_full_pipeline(pipeline_state)
            self.after(0, self.pipeline_finished, result_state, False)
        except Exception as e:
            self.after(0, self.pipeline_finished, str(e), True)

    def pipeline_finished(self, result_state, is_error):
        if is_error:
            result = str(result_state) # In case of error, result_state is the error string
            print(f"\n--- PIPELINE FAILED ---\nError: {result}\n")
            messagebox.showerror("Pipeline Failed", f"An error occurred: {result}")
        else:
            result = result_state.get("final_message", "Pipeline completed successfully!")
            print(f"\n--- PIPELINE FINISHED ---\n{result}\n")
            
            # --- NEW: Display Charts ---
            if result_state.get("chart_images"):
                print("Displaying charts...")
                for chart_img in result_state["chart_images"]:
                    ctk_img = ctk.CTkImage(light_image=chart_img, size=(chart_img.width, chart_img.height))
                    img_label = ctk.CTkLabel(self.charts_frame, image=ctk_img, text="")
                    img_label.pack(pady=10, padx=10)
                # Switch to charts tab
                self.results_tab_view.set("Charts")
            
            # Add download button
            self.download_button = ctk.CTkButton(self.control_frame, text="⬇️ Show Results Folder", fg_color="green", command=self.open_results)
            self.download_button.grid(row=2, column=0, padx=0, pady=10, sticky="ew")

        self.start_button.configure(state="normal", text="🚀 Start AutoML Pipeline")
        self.update_step_highlight(-1)

    def open_results(self):
        results_dir = os.path.join(os.getcwd(), "results")
        messagebox.showinfo("Results", f"Model and charts are saved in:\n{results_dir}")
        try:
            if sys.platform == "win32":
                os.startfile(results_dir)
            elif sys.platform == "darwin":
                os.system(f"open {results_dir}")
            else:
                os.system(f"xdg-open {results_dir}")
        except Exception as e:
            print(f"Could not open results folder: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()