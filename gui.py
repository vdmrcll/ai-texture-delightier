import os
import threading
import webbrowser
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import Menu, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image

from engine import DelighterInferenceEngine
try:
    from viewport import OPENGL_AVAILABLE, TextureViewport
except Exception:
    OPENGL_AVAILABLE = False
    TextureViewport = None


MODEL_FILENAME = "delighter_model_fp16.onnx"
MODEL_LABEL = "FP16"


class CTkApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class DelighterGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = CTkApp()
        self._configure_dpi_scaling()
        self.root.title("Texture Delighter | Beta")
        self.root.geometry("1024x720")
        self.root.minsize(880, 620)
        self.root.resizable(True, True)
        self.viewport_fov = ctk.IntVar(value=50)
        self.inference_mode = tk.StringVar(value="tiled")
        self._build_menu()

        self.engine = None
        self.last_output_path = None
        self.model_path = ctk.StringVar()
        self._engine_load_id = 0
        self.preview_texture = ctk.StringVar(value="lit")
        self.viewport = None
        self.viewport_error = None
        self.paths = {
            "lit": ctk.StringVar(),
            "normal": ctk.StringVar(),
            "ao": ctk.StringVar(),
            "mask": ctk.StringVar(),
            "output_dir": ctk.StringVar()
        }

        self._build_ui()
        self._load_engine_async()

    def _configure_dpi_scaling(self):
        """Allow CustomTkinter to handle standard OS scaling natively."""
        try:
            dpi = float(self.root.winfo_fpixels("1i"))
            system_scale = max(1.0, dpi / 96.0)
        except (AttributeError, tk.TclError, TypeError, ValueError):
            system_scale = 1.0

        ctk.set_widget_scaling(0.85)
        ctk.set_window_scaling(0.88)

    def _load_engine_async(self):
        model_name = os.path.join("weights", MODEL_FILENAME)
        self._engine_load_id += 1
        load_id = self._engine_load_id
        self.engine = None
        self.process_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.status_label.configure(text=f"Status: Loading {MODEL_LABEL}...", text_color="#FFC107")
        self._append_log(f"Initializing {MODEL_LABEL} engine...")

        def _init():
            try:
                engine = DelighterInferenceEngine(model_name)
                self.root.after(0, lambda: self._on_engine_ready(engine, load_id))
            except Exception as e:
                error = str(e)
                self.root.after(0, lambda error=error: self._on_engine_error(error, load_id))

        threading.Thread(target=_init, daemon=True).start()

    def _on_engine_ready(self, engine, load_id):
        if load_id != self._engine_load_id:
            return
        self.engine = engine
        device_str = engine.device_info
        self.process_btn.configure(state="normal")
        self.hw_label.configure(
            text=engine.device_display,
            text_color="#4CAF50" if engine.is_accelerated else "#FF9800",
        )
        self.precision_label.configure(text=f"Model: {MODEL_LABEL}")
        self.status_label.configure(text=f"Status: Ready ({MODEL_LABEL})", text_color="#4CAF50")
        self._append_log(f"Ready [{device_str}]")

    def _on_engine_error(self, error, load_id):
        if load_id != self._engine_load_id:
            return
        self.process_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Failed", text_color="#F44336")
        self._append_log(f"Init error: {error}")

    def _append_log(self, text):
        self.console_textbox.configure(state="normal")
        self.console_textbox.insert("end", text + "\n")
        self.console_textbox.see("end")
        self.console_textbox.configure(state="disabled")

    def _build_menu(self):
        menu_colors = {
            "tearoff": False,
            "font": ("Arial", 12),
            "background": "#151B24",
            "foreground": "#F2F2F2",
            "activebackground": "#1F6AA5",
            "activeforeground": "#FFFFFF",
            "disabledforeground": "#6B7280",
            "borderwidth": 1,
            "relief": "flat",
        }
        file_menu = Menu(self.root, **menu_colors)
        file_menu.add_command(label="Open model...", command=lambda: self._browse_file_or_dir("model"))
        file_menu.add_command(label="Open lit texture...", command=lambda: self._browse_file_or_dir("lit"))
        file_menu.add_separator()
        file_menu.add_command(label="Choose output directory...", command=lambda: self._browse_file_or_dir("output_dir"))
        file_menu.add_separator()
        preferences_menu = Menu(file_menu, **menu_colors)
        fov_menu = Menu(preferences_menu, **menu_colors)
        for fov in (30, 40, 50):
            fov_menu.add_radiobutton(
                label=f"{fov}°",
                variable=self.viewport_fov,
                value=fov,
                command=lambda value=fov: self._set_viewport_fov(value),
            )
        preferences_menu.add_cascade(label="Viewport FOV", menu=fov_menu)
        inference_menu = Menu(preferences_menu, **menu_colors)
        inference_menu.add_radiobutton(
            label="Full (High VRAM)",
            variable=self.inference_mode,
            value="full",
            command=lambda: self._set_inference_mode("full"),
        )
        inference_menu.add_radiobutton(
            label="Tiled (Low VRAM)",
            variable=self.inference_mode,
            value="tiled",
            command=lambda: self._set_inference_mode("tiled"),
        )
        preferences_menu.add_cascade(label="Inference mode", menu=inference_menu)
        file_menu.add_cascade(label="Preferences", menu=preferences_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        view_menu = Menu(self.root, **menu_colors)
        view_menu.add_command(label="Reset 3D view", command=self._reset_view)
        view_menu.add_command(label="2D texture preview", command=self._open_preview_window)

        help_menu = Menu(self.root, **menu_colors)
        help_menu.add_command(label="About", command=self._show_about)

        # Windows can render the native Tk menubar with its own light theme.
        # Use an in-window menubar so it matches the dark submenus.
        self.menu_bar = tk.Frame(self.root, height=30, bg="#151B24")
        self.menu_bar.grid(row=0, column=0, sticky="ew")
        self.menu_bar.grid_propagate(False)
        self.menu_bar.grid_columnconfigure(3, weight=1)
        for column, (label, submenu) in enumerate(
            (("File", file_menu), ("View", view_menu), ("Help", help_menu))
        ):
            menu_button = tk.Button(
                self.menu_bar,
                text=label,
                width=8,
                height=1,
                anchor="w",
                bg="#151B24",
                fg="#F2F2F2",
                activebackground="#1F6AA5",
                activeforeground="#FFFFFF",
                relief="flat",
                bd=0,
                highlightthickness=0,
                font=("Arial", 10),
                padx=10,
            )
            menu_button.configure(command=lambda menu=submenu, button=menu_button: self._show_popup_menu(menu, button))
            menu_button.grid(row=0, column=column, sticky="nsw", padx=(4 if column == 0 else 0, 0))

    @staticmethod
    def _show_popup_menu(menu, button):
        try:
            menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        finally:
            menu.grab_release()

    def _show_about(self):
        about_win = ctk.CTkToplevel(self.root)
        about_win.title("About Texture Delighter")
        about_win.geometry("430x255")
        about_win.resizable(False, False)
        about_win.transient(self.root)
        about_win.grab_set()

        ctk.CTkLabel(
            about_win,
            text="Texture Delighter",
            font=("Arial", 20, "bold"),
        ).pack(pady=(24, 6))
        ctk.CTkLabel(
            about_win,
            text="Generate a de-lighted albedo and inspect it on your OBJ model.",
            text_color="#9AA4B2",
            wraplength=360,
        ).pack(pady=(0, 14))
        ctk.CTkLabel(
            about_win,
            text="Developer: Vida Marcell",
            font=("Arial", 11, "bold"),
        ).pack(pady=(0, 10))

        repo_url = "https://github.com/vdmrcll/ai-texture-delightier"
        repo_link = ctk.CTkLabel(
            about_win,
            text=repo_url,
            text_color="#4EA1FF",
            cursor="hand2",
            font=("Arial", 10, "underline"),
        )
        repo_link.pack(pady=(0, 18))
        repo_link.bind("<Button-1>", lambda _event: webbrowser.open_new(repo_url))

        ctk.CTkButton(about_win, text="Close", width=90, command=about_win.destroy).pack()

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        header_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 6))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header_frame, text="Texture workspace", font=("Arial", 13, "bold"), text_color="#9AA4B2").grid(row=0, column=0, sticky="w")

        workspace = ctk.CTkFrame(self.root, fg_color="transparent")
        workspace.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 10))
        workspace.grid_columnconfigure(0, weight=2)
        workspace.grid_columnconfigure(1, weight=5)
        workspace.grid_rowconfigure(0, weight=1)

        input_frame = ctk.CTkFrame(workspace)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_frame = ctk.CTkFrame(workspace, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=0)

        viewport_frame = ctk.CTkFrame(right_frame)
        viewport_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        viewport_frame.grid_columnconfigure(0, weight=1)
        viewport_frame.grid_rowconfigure(1, weight=1)
        viewport_header = ctk.CTkFrame(viewport_frame, fg_color="transparent")
        viewport_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 8))
        ctk.CTkLabel(viewport_header, text="3D preview", font=("Arial", 15, "bold")).pack(side="left")
        ctk.CTkLabel(viewport_header, text="Drag to orbit · scroll to zoom", font=("Arial", 11), text_color="#9AA4B2").pack(side="left", padx=14)
        ctk.CTkButton(viewport_header, text="Reset view", width=88, height=28, command=self._reset_view).pack(side="right")
        self.preview_btn = ctk.CTkButton(viewport_header, text="2D Preview", width=88, height=28, state="disabled", command=self._open_preview_window)
        self.preview_btn.pack(side="right", padx=(0, 8))
        if OPENGL_AVAILABLE and TextureViewport is not None:
            try:
                self.viewport = TextureViewport(viewport_frame, highlightthickness=0, bd=0)
                self.viewport.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
            except Exception as error:
                self.viewport_error = str(error)
        if self.viewport is None:
            ctk.CTkLabel(viewport_frame, text="3D viewport unavailable. Check the execution log for the OpenGL error.", text_color="#FF9800").grid(row=1, column=0, sticky="nsew")

        input_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(input_frame, text="Inputs", font=("Arial", 15, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(input_frame, text="Drop files onto a field or browse to select them.", font=("Arial", 11), text_color="#9AA4B2", anchor="w").grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))

        slots = [
            ("Imported OBJ model", "model", "Path to OBJ mesh..."),
            ("Lit RGB map", "lit", "Path to lit texture..."),
            ("Normal map", "normal", "Path to normal map..."),
            ("Ambient occlusion", "ao", "Path to ambient occlusion map..."),
            ("UV / geometry mask", "mask", "Path to geometry mask..."),
            ("Output directory", "output_dir", "Default: source directory"),
        ]
        for index, (label_text, key, placeholder) in enumerate(slots):
            row_number = 2 + index
            row = ctk.CTkFrame(input_frame, fg_color="transparent")
            row.grid(row=row_number, column=0, sticky="ew", padx=14, pady=(0, 4))
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=label_text, anchor="w", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
            variable = self.model_path if key == "model" else self.paths[key]
            entry = ctk.CTkEntry(row, textvariable=variable, placeholder_text=placeholder, height=28)
            entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind('<<Drop>>', lambda e, k=key: self._handle_drop(e, k))
            ctk.CTkButton(row, text="Browse", width=65, height=28, command=lambda k=key: self._browse_file_or_dir(k)).grid(row=1, column=1)

        texture_choice = ctk.CTkFrame(input_frame, fg_color="transparent")
        texture_choice.grid(row=8, column=0, sticky="ew", padx=20, pady=(0, 8))
        ctk.CTkLabel(texture_choice, text="Preview texture", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self.lit_radio = ctk.CTkRadioButton(texture_choice, text="Lit", variable=self.preview_texture, value="lit", command=self._refresh_viewport)
        self.lit_radio.pack(side="left", padx=(0, 16))
        self.albedo_radio = ctk.CTkRadioButton(texture_choice, text="De-lighted", variable=self.preview_texture, value="albedo", state="disabled", command=self._refresh_viewport)
        self.albedo_radio.pack(side="left")

        log_frame = ctk.CTkFrame(right_frame, height=132)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_propagate(False)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_frame, text="Execution log", font=("Arial", 12, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=14, pady=(8, 4))
        self.console_textbox = ctk.CTkTextbox(log_frame, height=76, font=("Consolas", 9), fg_color="#0B1016", text_color="#52E69A", corner_radius=8)
        self.console_textbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.console_textbox.configure(state="disabled")
        if self.viewport_error:
            self._append_log(f"3D viewport initialization failed: {self.viewport_error}")
        elif self.viewport is not None:
            self._append_log("3D viewport ready [OpenGL / unlit texture mode]")

        action_frame = ctk.CTkFrame(self.root, fg_color="#151B24", corner_radius=0)
        action_frame.grid(row=3, column=0, sticky="ew")
        action_frame.grid_columnconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(action_frame, text="Status: Initializing...", font=("Arial", 10, "bold"), text_color="#FFC107")
        self.status_label.grid(row=0, column=0, sticky="w", padx=(16, 12), pady=8)
        self.hw_label = ctk.CTkLabel(action_frame, text="Device: Initializing...", font=("Arial", 10), text_color="#9AA4B2")
        self.hw_label.grid(row=0, column=1, sticky="w", pady=10)
        self.precision_label = ctk.CTkLabel(action_frame, text="Model: FP16", font=("Arial", 10), text_color="#9AA4B2")
        self.precision_label.grid(row=0, column=2, sticky="e", padx=(20, 20), pady=10)

        self.process_btn = ctk.CTkButton(
            action_frame, 
            text="Generate Albedo", 
            font=("Arial", 12, "bold"), 
            height=38,
            width=156,
            command=self._start_processing
        )
        self.process_btn.grid(row=0, column=3, sticky="e", padx=(0, 16), pady=6)

    def _validate_resolution(self, filepath):
        try:
            with Image.open(filepath) as img:
                w, h = img.size
                if w != 2048 or h != 2048:
                    self._append_log(f"WARNING: {os.path.basename(filepath)} is {w}x{h} (Model expects 2048x2048). Expect artifacts.")
        except Exception:
            pass

    def _reset_view(self):
        if self.viewport:
            self.viewport.reset_view()

    def _set_viewport_fov(self, fov):
        self.viewport_fov.set(int(fov))
        if self.viewport:
            self.viewport.set_fov(fov)

    def _set_inference_mode(self, mode):
        self.inference_mode.set(mode)
        label = "full resolution" if mode == "full" else "1024px tiled"
        self._append_log(f"Inference mode: {label}")

    def _refresh_viewport(self):
        if not self.viewport:
            return
        model = self.model_path.get()
        texture = self.paths["lit"].get() if self.preview_texture.get() == "lit" else self.last_output_path
        try:
            if model and os.path.isfile(model):
                self.viewport.load_mesh(model)
            if texture and os.path.isfile(texture):
                self.viewport.load_texture(texture)
        except Exception as error:
            self._append_log(f"3D preview error: {error}")

    def _set_model_path(self, path):
        if not path.lower().endswith(".obj"):
            messagebox.showwarning("Model Error", "The 3D preview currently supports OBJ meshes only.")
            return
        self.model_path.set(path)
        self._append_log(f"Loaded MODEL: {os.path.basename(path)}")
        self._refresh_viewport()

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
            if key == "model":
                self._set_model_path(clean_path)
                return
            self.paths[key].set(clean_path)
            self._append_log(f"Loaded {key.upper()}: {os.path.basename(clean_path)}")
            if key != "output_dir":
                self._validate_resolution(clean_path)
            if key == "lit":
                self._auto_fill_other_maps(clean_path)
                self._refresh_viewport()

    def _browse_file_or_dir(self, key):
        if key == "model":
            filename = filedialog.askopenfilename(filetypes=[("OBJ Mesh", "*.obj")])
            if filename:
                self._set_model_path(filename)
            return
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
                    self._refresh_viewport()

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
        self.status_label.configure(text="Status: Processing...", text_color="#2196F3")

        def log_ui(msg):
            self.root.after(0, lambda: self._append_log(msg))

        def _work():
            try:
                out_file, dims, dev = self.engine.process_texture(
                    lit,
                    norm,
                    ao,
                    mask,
                    output_path,
                    log_callback=log_ui,
                    tiled_mode=self.inference_mode.get() == "tiled",
                )
                self.root.after(0, lambda: self._on_success(out_file, dims, dev))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_work, daemon=True).start()

    def _on_success(self, out_file, dims, dev):
        self.process_btn.configure(state="normal", text="Generate Albedo")
        self.preview_btn.configure(state="normal")
        self.albedo_radio.configure(state="normal")
        self.last_output_path = out_file
        self.status_label.configure(text="Status: Completed", text_color="#4CAF50")
        self._append_log(f"Done. Saved to {out_file}\n")
        self.preview_texture.set("albedo")
        self._refresh_viewport()

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
        preview_win.geometry("700x580")
        preview_win.minsize(700, 580)
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
            img_tk = ctk.CTkImage(light_image=blended, dark_image=blended, size=blended.size)
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
