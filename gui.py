import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk

from engine import DelighterInferenceEngine

class CTkApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class DelighterGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = CTkApp()
        self.root.title("AI Texture De-Lighter | v0.1 Beta-test")
        self.root.geometry("520x670")
        self.root.resizable(False, False)

        self.engine = None
        self.last_output_path = None
        self.paths = {
            "lit": ctk.StringVar(),
            "normal": ctk.StringVar(),
            "ao": ctk.StringVar(),
            "mask": ctk.StringVar(),
            "output_dir": ctk.StringVar()
        }

        self._build_ui()
        self._load_engine_async()

    def _load_engine_async(self):
        def _init():
            try:
                self._append_log("Initializing engine...")
                self.engine = DelighterInferenceEngine("delighter_model_jit.pt")
                device_str = self.engine.device_info
                self.hw_label.configure(text=f"Device: {device_str}", text_color="#4CAF50" if "GPU" in device_str else "#FF9800")
                self.status_label.configure(text="Status: Ready", text_color="#4CAF50")
                self._append_log(f"Ready [{device_str}]")
            except Exception as e:
                self.status_label.configure(text="Status: Failed", text_color="#F44336")
                self._append_log(f"Init error: {e}")
        
        threading.Thread(target=_init, daemon=True).start()

    def _append_log(self, text):
        self.console_textbox.configure(state="normal")
        self.console_textbox.insert("end", text + "\n")
        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")

    def _build_ui(self):
        header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 2))

        title = ctk.CTkLabel(header_frame, text="AI Texture De-Lighter", font=("Arial", 16, "bold"))
        title.pack(side="left")

        self.hw_label = ctk.CTkLabel(header_frame, text="Device: Initializing...", font=("Arial", 10, "bold"), text_color="gray")
        self.hw_label.pack(side="right")

        frame = ctk.CTkFrame(self.root)
        frame.pack(padx=15, pady=4, fill="x")

        slots = [
            ("Lit RGB Map:", "lit", "Path to lit texture..."),
            ("Normal Map:", "normal", "Path to normal map..."),
            ("Ambient Occlusion:", "ao", "Path to ambient occlusion map..."),
            ("UV / Mask:", "mask", "Path to geometry mask..."),
            ("Output Directory:", "output_dir", "Default: Source directory"),
        ]

        for label_text, key, placeholder in slots:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)

            lbl = ctk.CTkLabel(row, text=label_text, anchor="w", font=("Arial", 11, "bold"))
            lbl.pack(anchor="w", padx=2, pady=(2, 0))

            sub_row = ctk.CTkFrame(row, fg_color="transparent")
            sub_row.pack(fill="x", pady=(1, 2))

            entry = ctk.CTkEntry(sub_row, textvariable=self.paths[key], placeholder_text=placeholder, width=405, height=26)
            entry.pack(side="left", padx=(0, 6))

            entry.drop_target_register(DND_FILES)
            entry.dnd_bind('<<Drop>>', lambda e, k=key: self._handle_drop(e, k))

            btn = ctk.CTkButton(sub_row, text="Browse", width=60, height=26, command=lambda k=key: self._browse_file_or_dir(k))
            btn.pack(side="left")

        log_frame = ctk.CTkFrame(self.root)
        log_frame.pack(padx=15, pady=6, fill="both", expand=True)

        log_title = ctk.CTkLabel(log_frame, text="Execution Log", font=("Arial", 11, "bold"), anchor="w")
        log_title.pack(anchor="w", padx=8, pady=(3, 0))

        self.console_textbox = ctk.CTkTextbox(log_frame, font=("Consolas", 10), fg_color="#0a0a0a", text_color="#00E676", height=80)
        self.console_textbox.pack(padx=8, pady=4, fill="both", expand=True)
        self.console_textbox.configure(state="disabled")

        action_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        action_frame.pack(fill="x", padx=15, pady=(2, 4))

        self.status_label = ctk.CTkLabel(action_frame, text="Status: Initializing...", font=("Arial", 11, "bold"), text_color="#FFC107")
        self.status_label.pack(side="left")

        self.preview_btn = ctk.CTkButton(
            action_frame,
            text="Preview",
            font=("Arial", 11, "bold"),
            height=34,
            width=90,
            fg_color="#37474F",
            hover_color="#455A64",
            state="disabled",
            command=self._open_preview_window
        )
        self.preview_btn.pack(side="right", padx=(6, 0))

        self.process_btn = ctk.CTkButton(
            action_frame, 
            text="Generate Albedo", 
            font=("Arial", 12, "bold"), 
            height=34, 
            width=140,
            command=self._start_processing
        )
        self.process_btn.pack(side="right")

    def _validate_resolution(self, filepath):
        if key_is_dir := (filepath == self.paths["output_dir"].get()):
            return
        try:
            with Image.open(filepath) as img:
                w, h = img.size
                if w != 2048 or h != 2048:
                    self._append_log(f"WARNING: {os.path.basename(filepath)} is {w}x{h} (Model expects 2048x2048). Expect artifacts.")
        except Exception:
            pass

    def _handle_drop(self, event, key):
        files = self.root.tk.splitlist(event.data)
        if files:
            clean_path = files[0].strip('{}')
            self.paths[key].set(clean_path)
            self._append_log(f"Loaded {key.upper()}: {os.path.basename(clean_path)}")
            if key != "output_dir":
                self._validate_resolution(clean_path)
            if key == "lit":
                self._auto_fill_other_maps(clean_path)

    def _browse_file_or_dir(self, key):
        if key == "output_dir":
            folder = filedialog.askdirectory()
            if folder:
                self.paths[key].set(folder)
                self._append_log(f"Output set: {folder}")
        else:
            filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff *.exr")])
            if filename:
                self.paths[key].set(filename)
                self._append_log(f"Loaded {key.upper()}: {os.path.basename(filename)}")
                self._validate_resolution(filename)
                if key == "lit":
                    self._auto_fill_other_maps(filename)

    def _auto_fill_other_maps(self, lit_path):
        folder, filename = os.path.split(lit_path)
        
        if not self.paths["output_dir"].get():
            self.paths["output_dir"].set(folder)
            self._append_log(f"Output dir set: {folder}")

        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            f_lower = file.lower()
            if "normal" in f_lower and not self.paths["normal"].get():
                self.paths["normal"].set(full_path)
                self._append_log(f"Matched Normal: {file}")
                self._validate_resolution(full_path)
            elif "ao" in f_lower and not self.paths["ao"].get():
                self.paths["ao"].set(full_path)
                self._append_log(f"Matched AO: {file}")
                self._validate_resolution(full_path)
            elif "mask" in f_lower and not self.paths["mask"].get():
                self.paths["mask"].set(full_path)
                self._append_log(f"Matched Mask: {file}")
                self._validate_resolution(full_path)

    def _start_processing(self):
        lit = self.paths["lit"].get()
        norm = self.paths["normal"].get()
        ao = self.paths["ao"].get()
        mask = self.paths["mask"].get()

        if not all([lit, norm, ao, mask]):
            messagebox.showwarning("Validation Error", "All four required input maps must be specified.")
            return

        if not self.engine:
            messagebox.showerror("State Error", "Inference engine is not initialized.")
            return

        out_dir = self.paths["output_dir"].get()
        if not out_dir:
            out_dir = os.path.dirname(lit)

        output_path = os.path.join(out_dir, "delighted_albedo.png")

        self.process_btn.configure(state="disabled", text="Processing...")
        self.preview_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Processing...", text_color="#2196F3")

        def log_ui(msg):
            self.root.after(0, lambda: self._append_log(msg))

        def _work():
            try:
                out_file, dims, dev = self.engine.process_texture(
                    lit, norm, ao, mask, output_path, log_callback=log_ui
                )
                self.root.after(0, lambda: self._on_success(out_file, dims, dev))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_work, daemon=True).start()

    def _on_success(self, out_file, dims, dev):
        self.process_btn.configure(state="normal", text="Generate Albedo")
        self.preview_btn.configure(state="normal")
        self.last_output_path = out_file
        self.status_label.configure(text="Status: Completed", text_color="#4CAF50")
        self._append_log(f"Done. Saved to {out_file}\n")

    def _on_error(self, err_msg):
        self.process_btn.configure(state="normal", text="Generate Albedo")
        self.status_label.configure(text="Status: Failed", text_color="#F44336")
        self._append_log(f"Error: {err_msg}\n")
        messagebox.showerror("Execution Error", f"Inference failed:\n{err_msg}")

    def _open_preview_window(self):
        if not self.last_output_path or not os.path.exists(self.last_output_path):
            messagebox.showwarning("Preview Error", "No generated output file found.")
            return

        lit_path = self.paths["lit"].get()
        if not lit_path or not os.path.exists(lit_path):
            messagebox.showwarning("Preview Error", "Original lit map is missing.")
            return

        preview_win = ctk.CTkToplevel(self.root)
        preview_win.title("Texture Comparison Viewer")
        preview_win.geometry("700x500")
        preview_win.grab_set()

        try:
            lit_img = Image.open(lit_path).convert("RGB")
            out_img = Image.open(self.last_output_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images for preview:\n{e}")
            preview_win.destroy()
            return

        display_size = (420, 420)
        lit_img.thumbnail(display_size, Image.Resampling.LANCZOS)
        out_img.thumbnail(display_size, Image.Resampling.LANCZOS)

        img_container = ctk.CTkFrame(preview_win, fg_color="transparent")
        img_container.pack(padx=20, pady=20, expand=True)

        lbl_image = ctk.CTkLabel(img_container, text="")
        lbl_image.pack(pady=5)

        slider_frame = ctk.CTkFrame(preview_win, fg_color="transparent")
        slider_frame.pack(fill="x", padx=40, pady=(0, 20))

        info_lbl = ctk.CTkLabel(slider_frame, text="Comparison Mode: Drag to slide blend", font=("Arial", 11, "bold"), text_color="gray")
        info_lbl.pack(pady=(0, 5))

        def update_blend(val):
            ratio = float(val) / 100.0
            blended = Image.blend(lit_img, out_img, alpha=ratio)
            img_tk = ImageTk.PhotoImage(blended)
            lbl_image.configure(image=img_tk)
            lbl_image.image = img_tk
            
            if ratio < 0.3:
                info_lbl.configure(text="Showing: Original Lit Texture")
            elif ratio > 0.7:
                info_lbl.configure(text="Showing: Delighted Albedo Output")
            else:
                info_lbl.configure(text=f"Blending: {int(ratio*100)}% Albedo")

        slider = ctk.CTkSlider(slider_frame, from_=0, to=100, command=update_blend, width=500)
        slider.pack()
        slider.set(100)
        update_blend(100)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DelighterGUI()
    app.run()