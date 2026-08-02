import os
import threading
import cv2
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
        self.root.title("AI Texture De-Lighter | Beta")
        self.root.geometry("920x700")
        self.root.minsize(820, 620)
        self.root.resizable(True, True)

        self.engine = None
        self.last_output_path = None
        self.precision = ctk.StringVar(value="FP16")
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
        selected_precision = self.precision.get()
        model_name = "weights/delighter_model_fp16.onnx" if selected_precision == "FP16" else "weights/delighter_model_fp32.onnx"
        self.engine = None
        self.process_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.status_label.configure(text=f"Status: Loading {selected_precision}...", text_color="#FFC107")
        self._append_log("Initializing engine...")

        def _init():
            try:
                engine = DelighterInferenceEngine(model_name)
                self.root.after(0, lambda: self._on_engine_ready(engine))
            except Exception as e:
                error = str(e)
                self.root.after(0, lambda error=error: self._on_engine_error(error))

        threading.Thread(target=_init, daemon=True).start()

    def _on_precision_change(self, value):
        self._append_log(f"Precision selected: {value}")
        self._load_engine_async()

    def _on_engine_ready(self, engine):
        self.engine = engine
        device_str = engine.device_info
        self.precision_menu.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.hw_label.configure(
            text=engine.device_display,
            text_color="#4CAF50" if engine.is_accelerated else "#FF9800",
        )
        self.status_label.configure(text=f"Status: Ready ({self.precision.get()})", text_color="#4CAF50")
        self._append_log(f"Ready [{device_str}]")

    def _on_engine_error(self, error):
        self.precision_menu.configure(state="normal")
        self.process_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Failed", text_color="#F44336")
        self._append_log(f"Init error: {error}")

    def _append_log(self, text):
        self.console_textbox.configure(state="normal")
        self.console_textbox.insert("end", text + "\n")
        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 14))
        header_frame.grid_columnconfigure(0, weight=1)

        heading = ctk.CTkFrame(header_frame, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="w")
        title = ctk.CTkLabel(heading, text="AI Texture De-Lighter", font=("Arial", 24, "bold"))
        title.pack(anchor="w")
        subtitle = ctk.CTkLabel(heading, text="Remove baked lighting and generate a clean albedo map", font=("Arial", 12), text_color="#9AA4B2")
        subtitle.pack(anchor="w", pady=(3, 0))

        controls = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls.grid(row=0, column=1, sticky="e")
        self.hw_label = ctk.CTkLabel(controls, text="Device: Initializing...", font=("Arial", 11, "bold"), text_color="#9AA4B2")
        self.hw_label.grid(row=0, column=0, columnspan=2, sticky="e", pady=(0, 8))
        precision_label = ctk.CTkLabel(controls, text="Model precision", font=("Arial", 11, "bold"))
        precision_label.grid(row=1, column=0, padx=(0, 10))
        self.precision_menu = ctk.CTkOptionMenu(controls, values=["FP16", "FP32"], variable=self.precision, command=self._on_precision_change, width=92, height=30)
        self.precision_menu.grid(row=1, column=1)

        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 18))
        content.grid_columnconfigure(0, weight=3, uniform="content")
        content.grid_columnconfigure(1, weight=2, uniform="content")
        content.grid_rowconfigure(0, weight=1)

        input_frame = ctk.CTkFrame(content)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(input_frame, text="Texture inputs", font=("Arial", 15, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=20, pady=(18, 2))
        ctk.CTkLabel(input_frame, text="Drop files onto a field or browse to select them.", font=("Arial", 11), text_color="#9AA4B2", anchor="w").grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))

        slots = [
            ("Lit RGB map", "lit", "Path to lit texture..."),
            ("Normal map", "normal", "Path to normal map..."),
            ("Ambient occlusion", "ao", "Path to ambient occlusion map..."),
            ("UV / geometry mask", "mask", "Path to geometry mask..."),
            ("Output directory", "output_dir", "Default: source directory"),
        ]
        for index, (label_text, key, placeholder) in enumerate(slots, start=2):
            row = ctk.CTkFrame(input_frame, fg_color="transparent")
            row.grid(row=index, column=0, sticky="ew", padx=20, pady=(0, 11))
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=label_text, anchor="w", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
            entry = ctk.CTkEntry(row, textvariable=self.paths[key], placeholder_text=placeholder, height=34)
            entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind('<<Drop>>', lambda e, k=key: self._handle_drop(e, k))
            ctk.CTkButton(row, text="Browse", width=82, height=34, command=lambda k=key: self._browse_file_or_dir(k)).grid(row=1, column=1)

        log_frame = ctk.CTkFrame(content)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(log_frame, text="Execution log", font=("Arial", 15, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=18, pady=(18, 2))
        ctk.CTkLabel(log_frame, text="Engine messages and processing details", font=("Arial", 11), text_color="#9AA4B2", anchor="w").grid(row=1, column=0, sticky="w", padx=18, pady=(0, 12))
        self.console_textbox = ctk.CTkTextbox(log_frame, font=("Consolas", 10), fg_color="#0B1016", text_color="#52E69A", corner_radius=8)
        self.console_textbox.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.console_textbox.configure(state="disabled")

        action_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        action_frame.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 24))

        self.status_label = ctk.CTkLabel(action_frame, text="Status: Initializing...", font=("Arial", 11, "bold"), text_color="#FFC107")
        self.status_label.pack(side="left")

        self.preview_btn = ctk.CTkButton(
            action_frame,
            text="Preview",
            font=("Arial", 11, "bold"),
            height=38,
            width=100,
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
            height=38,
            width=156,
            command=self._start_processing
        )
        self.process_btn.pack(side="right")

    def _validate_resolution(self, filepath):
        try:
            with Image.open(filepath) as img:
                w, h = img.size
                if w != 2048 or h != 2048:
                    self._append_log(f"WARNING: {os.path.basename(filepath)} is {w}x{h} (Model expects 2048x2048). Expect artifacts.")
        except Exception:
            pass

    def _validate_inputs(self, paths):
        sizes = []
        for label, path in paths:
            if not os.path.isfile(path):
                raise ValueError(f"{label} does not exist: {path}")
            try:
                with Image.open(path) as img:
                    size = img.size
            except (OSError, ValueError):
                # Pillow may not have an EXR plugin; OpenCV can still inspect
                # the dimensions and the inference engine supports float maps.
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Unable to read {label}: {path}")
                size = (image.shape[1], image.shape[0])
            sizes.append((label, size))
        expected = sizes[0][1]
        mismatches = [(label, size) for label, size in sizes if size != expected]
        if mismatches:
            details = ", ".join(f"{label}={size[0]}x{size[1]}" for label, size in sizes)
            raise ValueError(f"All maps must have matching dimensions ({details})")

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

        try:
            self._validate_inputs([
                ("Lit map", lit), ("Normal map", norm),
                ("AO map", ao), ("Mask", mask),
            ])
        except (OSError, ValueError) as e:
            messagebox.showwarning("Validation Error", str(e))
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
        self.precision_menu.configure(state="disabled")
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
        self.precision_menu.configure(state="normal")
        self.preview_btn.configure(state="normal")
        self.last_output_path = out_file
        self.status_label.configure(text="Status: Completed", text_color="#4CAF50")
        self._append_log(f"Done. Saved to {out_file}\n")

    def _on_error(self, err_msg):
        self.process_btn.configure(state="normal", text="Generate Albedo")
        self.precision_menu.configure(state="normal")
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
