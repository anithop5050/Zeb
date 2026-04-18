"""
Watermark Studio
================
Modern dark-themed GUI for embedding and extracting watermarks.
"""

import os
import sys

# Setup path for imports
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_project_root = os.path.dirname(_current_dir)
# Ensure project root is in path
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import threading

# Import inference functions - use conditional import
try:
    # Try absolute import first (when run from root)
    from inference.inference import (
        load_models, load_image, save_image, generate_watermark,
        embed_watermark_tiled, extract_watermark, calculate_ber, calculate_psnr,
        WATERMARK_LEN, MODEL_SIZE, compute_texture_mask, compute_perceptual_mask_v2,
        compute_channel_weights, get_optimal_alpha
    )
except (ModuleNotFoundError, ImportError):
    # Fallback: load directly from sibling file (when run from inference dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference_module", os.path.join(_current_dir, "inference.py"))
    inference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_module)
    load_models = inference_module.load_models
    load_image = inference_module.load_image
    save_image = inference_module.save_image
    generate_watermark = inference_module.generate_watermark
    embed_watermark_tiled = inference_module.embed_watermark_tiled
    extract_watermark = inference_module.extract_watermark
    calculate_ber = inference_module.calculate_ber
    calculate_psnr = inference_module.calculate_psnr
    WATERMARK_LEN = inference_module.WATERMARK_LEN
    MODEL_SIZE = inference_module.MODEL_SIZE
    compute_texture_mask = inference_module.compute_texture_mask
    compute_perceptual_mask_v2 = inference_module.compute_perceptual_mask_v2
    compute_channel_weights = inference_module.compute_channel_weights
    get_optimal_alpha = inference_module.get_optimal_alpha

# Import seed registry
from core.seed_registry import SeedRegistry

# Import reliability module for round-trip hardening (R1.1-R1.8)
try:
    from core.reliability import (
        validate_save_path,
        clamp_alpha,
        load_image_exact,
        post_embed_verify,
        compute_ber as reliability_compute_ber,
        ALPHA_FLOOR,
        ALPHA_CEIL,
        logger as reliability_logger,
    )
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False


# Modern dark color scheme
COLORS = {
    'bg_dark': '#0d1117',
    'bg_medium': '#161b22',
    'bg_light': '#21262d',
    'bg_hover': '#30363d',
    'accent': '#58a6ff',
    'accent_hover': '#79b8ff',
    'success': '#3fb950',
    'warning': '#d29922',
    'error': '#f85149',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
    'border': '#30363d',
}


class ImageCard(tk.Frame):
    """Modern image preview card."""
    def __init__(self, parent, title, size=200, **kwargs):
        super().__init__(parent, bg=COLORS['bg_medium'], **kwargs)
        self.card_size = size
        
        # Title
        tk.Label(self, text=title, bg=COLORS['bg_medium'], 
                fg=COLORS['text_dim'], font=('Segoe UI', 10)).pack(pady=(10, 5))
        
        # Image container
        self.img_frame = tk.Frame(self, bg=COLORS['border'], padx=2, pady=2)
        self.img_frame.pack(padx=10, pady=5)
        
        self.canvas = tk.Canvas(self.img_frame, width=size, height=size, 
                               bg=COLORS['bg_dark'], highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(size//2, size//2, text="No Image", 
                               fill=COLORS['text_dim'], font=('Segoe UI', 10))
        
        # Size label
        self.size_label = tk.Label(self, text="", bg=COLORS['bg_medium'], 
                                  fg=COLORS['text_dim'], font=('Segoe UI', 9))
        self.size_label.pack(pady=(0, 10))
        
        self.photo = None
    
    def set_image(self, path):
        try:
            # Force a fresh decode each time so preview updates when a file is overwritten.
            with Image.open(path) as img_src:
                img = img_src.convert("RGB")
                orig_size = img.size
                img.thumbnail((self.card_size, self.card_size))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete('all')
            self.canvas.create_image(self.card_size//2, self.card_size//2, image=self.photo)
            self.size_label.config(text=f"{orig_size[0]} × {orig_size[1]}")
        except Exception as e:
            print(f"Preview error: {e}")
    
    def set_image_array(self, arr, label=""):
        """Set image from numpy array (H,W,3) in [0,255] or [0,1]."""
        try:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)
            orig_size = img.size
            img.thumbnail((self.card_size, self.card_size))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete('all')
            self.canvas.create_image(self.card_size//2, self.card_size//2, image=self.photo)
            self.size_label.config(text=label if label else f"{orig_size[0]} × {orig_size[1]}")
        except Exception as e:
            print(f"Preview array error: {e}")
    
    def clear(self):
        """Clear the image display."""
        self.canvas.delete('all')
        self.canvas.create_text(self.card_size//2, self.card_size//2, text="No Image", 
                               fill=COLORS['text_dim'], font=('Segoe UI', 10))
        self.size_label.config(text="")


class WatermarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ZEb:Robust Invisible Image Watermarking Using Deep Learning")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.minsize(900, 650)
        self.root.state('zoomed')
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mode = tk.StringVar(value='embed')
        self.alpha = tk.DoubleVar(value=0.035)  # Default matches auto-tune base alpha
        self.seed = tk.IntVar(value=42)
        self.selected_owner = tk.StringVar(value="")
        # Keep manual presets in control by default. Auto-tune is optional.
        self.auto_alpha_enabled = tk.BooleanVar(value=False)
        self.use_semantic = tk.BooleanVar(value=False)
        self.use_hvs = tk.BooleanVar(value=True)
        self.use_poison = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Starting up...")
        
        # Registry
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.registry = SeedRegistry(os.path.join(_project_root, "core", "seed_registry.db"))
        self.owners_list = self.load_owners_list()
        
        # Model variables
        self.device = None
        self.encoder = None
        self.decoder = None
        self.semantic_encoder = None
        self.semantic_decoder = None
        self.poisoner = None
        self.models_loaded = False
        self.is_processing = False
        self.img_tensor = None  # Cached input tensor for auto-tune
        
        # Full-resolution X-ray arrays for click-to-enlarge
        self.xray_full_combined = None
        self.xray_full_semantic = None
        self.xray_full_adversarial = None
        
        self.create_widgets()
        self.root.after(100, self.load_models_async)
    
    def load_owners_list(self) -> list:
        """Load list of owner names from registry."""
        try:
            seeds = self.registry.get_all_seeds()
            return [f"{s['owner_name']} (seed: {s['seed']})" for s in seeds]
        except:
            return []
    
    def on_owner_selected(self, event=None):
        """Auto-fill seed when owner is selected."""
        selected = self.selected_owner.get()
        if not selected:
            return
        
        # Extract seed from selected owner string
        try:
            seed_str = selected.split("seed: ")[-1].rstrip(")")
            seed = int(seed_str)
            self.seed.set(seed)
        except:
            pass
    
    def open_register_dialog(self):
        """Open dialog to register new owner."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Owner")
        dialog.geometry("460x520")
        dialog.minsize(420, 420)
        dialog.resizable(True, True)
        dialog.configure(bg=COLORS['bg_dark'])
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog over parent window.
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - (460 // 2)
        y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - (520 // 2)
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

        # Form frame
        form = tk.Frame(dialog, bg=COLORS['bg_dark'])
        form.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 10))
        
        # Name
        tk.Label(form, text="Owner Name *", bg=COLORS['bg_dark'], 
                fg=COLORS['text'], font=('Segoe UI', 10)).pack(anchor=tk.W, pady=(0, 5))
        name_entry = tk.Entry(form, bg=COLORS['bg_medium'], fg=COLORS['text'],
                             insertbackground=COLORS['text'], relief=tk.FLAT,
                             font=('Segoe UI', 10))
        name_entry.pack(fill=tk.X, ipady=8, pady=(0, 12))
        
        # Email
        tk.Label(form, text="Email Address *", bg=COLORS['bg_dark'], 
                fg=COLORS['text'], font=('Segoe UI', 10)).pack(anchor=tk.W, pady=(0, 5))
        email_entry = tk.Entry(form, bg=COLORS['bg_medium'], fg=COLORS['text'],
                              insertbackground=COLORS['text'], relief=tk.FLAT,
                              font=('Segoe UI', 10))
        email_entry.pack(fill=tk.X, ipady=8, pady=(0, 12))
        
        # Organization
        tk.Label(form, text="Organization", bg=COLORS['bg_dark'], 
                fg=COLORS['text'], font=('Segoe UI', 10)).pack(anchor=tk.W, pady=(0, 5))
        org_entry = tk.Entry(form, bg=COLORS['bg_medium'], fg=COLORS['text'],
                            insertbackground=COLORS['text'], relief=tk.FLAT,
                            font=('Segoe UI', 10))
        org_entry.pack(fill=tk.X, ipady=8, pady=(0, 12))
        
        # License type
        tk.Label(form, text="License Type", bg=COLORS['bg_dark'], 
                fg=COLORS['text'], font=('Segoe UI', 10)).pack(anchor=tk.W, pady=(0, 5))
        license_var = tk.StringVar(value="personal")
        license_combo = ttk.Combobox(form, textvariable=license_var,
                                    values=["personal", "non-exclusive", "exclusive"],
                                    state='readonly', font=('Segoe UI', 10))
        license_combo.pack(fill=tk.X, ipady=6, pady=(0, 15))
        
        # Buttons (fixed at bottom, always visible)
        btn_frame = tk.Frame(dialog, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        def register():
            name = name_entry.get().strip()
            email = email_entry.get().strip()
            org = org_entry.get().strip()
            license_type = license_var.get()
            
            if not name or not email:
                messagebox.showwarning("Required", "Name and email are required")
                return
            
            try:
                seed = self.registry.register_seed(name, email, org, license_type)
                self.owners_list = self.load_owners_list()
                self.owner_combo['values'] = self.owners_list
                self.selected_owner.set(f"{name} (seed: {seed})")
                self.seed.set(seed)
                messagebox.showinfo("Success", f"✅ Registered! Assigned seed: {seed}")
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e))
        
        tk.Button(btn_frame, text="Register", bg=COLORS['success'], fg='white',
                 font=('Segoe UI', 10, 'bold'), relief=tk.FLAT, padx=20,
                 command=register).pack(side=tk.RIGHT, padx=(8, 0))
        
        tk.Button(btn_frame, text="Cancel", bg=COLORS['bg_light'], fg=COLORS['text'],
                 font=('Segoe UI', 10), relief=tk.FLAT, padx=20,
                 command=dialog.destroy).pack(side=tk.RIGHT)
    
    def create_widgets(self):
        """Create modern UI."""
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(main, bg=COLORS['bg_dark'])
        header.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header, text="🛡️ZEb Watermark Studio", bg=COLORS['bg_dark'],
                fg=COLORS['text'], font=('Segoe UI', 22, 'bold')).pack(side=tk.LEFT)
        
        tk.Label(header, text=f"Model: {MODEL_SIZE}×{MODEL_SIZE} • 64-bit", 
                bg=COLORS['bg_dark'], fg=COLORS['text_dim'], 
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(15, 0), pady=(8, 0))
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        left = tk.Frame(content, bg=COLORS['bg_medium'], width=350)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)
        
        self.create_controls(left)
        
        # Right panel
        right = tk.Frame(content, bg=COLORS['bg_dark'])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_preview(right)
        
        # Status bar
        status = tk.Frame(main, bg=COLORS['bg_medium'], height=40)
        status.pack(fill=tk.X, pady=(15, 0))
        status.pack_propagate(False)
        
        tk.Label(status, textvariable=self.status_text, bg=COLORS['bg_medium'], 
                fg=COLORS['text_dim'], font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=15, pady=10)
        
        self.progress = ttk.Progressbar(status, mode='indeterminate', length=120)
        self.progress.pack(side=tk.RIGHT, padx=15, pady=10)
    
    def create_controls(self, parent):
        """Create control panel with scrolling."""
        pad = {'padx': 15}

        # Create process button FIRST - anchored at bottom for always-visible access
        self.process_btn = tk.Button(parent, text="[PROCESS] EMBED WATERMARK",
                                    bg=COLORS['accent'], fg='white',
                                    font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
                                    pady=12, command=self.process_image)
        self.process_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)

        # Create scrollable frame with proper sizing
        canvas = tk.Canvas(parent, bg=COLORS['bg_medium'], highlightthickness=0, width=300)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_medium'], width=300)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar at top, fill available space
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Mode buttons
        mode_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_medium'])
        mode_frame.pack(fill=tk.X, **pad, pady=(15, 8))

        tk.Label(mode_frame, text="MODE", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)

        btn_row = tk.Frame(mode_frame, bg=COLORS['bg_medium'])
        btn_row.pack(fill=tk.X, pady=(8, 0))

        self.embed_btn = tk.Button(btn_row, text="EMBED", bg=COLORS['accent'], fg='white',
                                  font=('Segoe UI', 10, 'bold'), relief=tk.FLAT, pady=8,
                                  command=lambda: self.set_mode('embed'))
        self.embed_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.decode_btn = tk.Button(btn_row, text="EXTRACT", bg=COLORS['bg_light'],
                                   fg=COLORS['text'], font=('Segoe UI', 10, 'bold'),
                                   relief=tk.FLAT, pady=8,
                                   command=lambda: self.set_mode('decode'))
        self.decode_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))

        # Separator
        tk.Frame(scrollable_frame, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=15, pady=8)

        # Input file
        file_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_medium'])
        file_frame.pack(fill=tk.X, **pad)

        tk.Label(file_frame, text="INPUT IMAGE", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)

        input_row = tk.Frame(file_frame, bg=COLORS['bg_medium'])
        input_row.pack(fill=tk.X, pady=(8, 0))

        self.input_entry = tk.Entry(input_row, textvariable=self.input_path,
                                   bg=COLORS['bg_dark'], fg=COLORS['text'],
                                   insertbackground=COLORS['text'], relief=tk.FLAT,
                                   font=('Segoe UI', 10))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))

        tk.Button(input_row, text="Browse", bg=COLORS['bg_light'], fg=COLORS['text'],
                 font=('Segoe UI', 9), relief=tk.FLAT, padx=10,
                 command=self.browse_input).pack(side=tk.RIGHT)

        # Output file
        self.output_frame = tk.Frame(file_frame, bg=COLORS['bg_medium'])
        self.output_frame.pack(fill=tk.X, pady=(12, 0))

        tk.Label(self.output_frame, text="OUTPUT IMAGE", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)

        output_row = tk.Frame(self.output_frame, bg=COLORS['bg_medium'])
        output_row.pack(fill=tk.X, pady=(8, 0))

        self.output_entry = tk.Entry(output_row, textvariable=self.output_path,
                                    bg=COLORS['bg_dark'], fg=COLORS['text'],
                                    insertbackground=COLORS['text'], relief=tk.FLAT,
                                    font=('Segoe UI', 10))
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))

        tk.Button(output_row, text="Browse", bg=COLORS['bg_light'], fg=COLORS['text'],
                 font=('Segoe UI', 9), relief=tk.FLAT, padx=10,
                 command=self.browse_output).pack(side=tk.RIGHT)

        # Separator
        tk.Frame(scrollable_frame, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=15, pady=8)

        # Owner selection
        owner_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_medium'])
        owner_frame.pack(fill=tk.X, **pad)

        tk.Label(owner_frame, text="WATERMARK OWNER", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)

        owner_row = tk.Frame(owner_frame, bg=COLORS['bg_medium'])
        owner_row.pack(fill=tk.X, pady=(8, 0))

        self.owner_combo = ttk.Combobox(owner_row, textvariable=self.selected_owner,
                                       values=self.owners_list, state='readonly',
                                       font=('Segoe UI', 9), width=18)
        self.owner_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))
        self.owner_combo.bind('<<ComboboxSelected>>', self.on_owner_selected)

        tk.Button(owner_row, text="Register", bg=COLORS['bg_light'], fg=COLORS['text'],
                 font=('Segoe UI', 8, 'bold'), relief=tk.FLAT, padx=8,
                 command=self.open_register_dialog).pack(side=tk.RIGHT)

        # Separator
        tk.Frame(scrollable_frame, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=15, pady=8)

        # Parameters
        self.param_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_medium'])
        self.param_frame.pack(fill=tk.X, **pad)

        tk.Label(self.param_frame, text="PARAMETERS", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)

        # ALPHA - Strength (IMPROVED UX)
        alpha_header = tk.Frame(self.param_frame, bg=COLORS['bg_medium'])
        alpha_header.pack(fill=tk.X, pady=(10, 0))

        tk.Label(alpha_header, text="INVISIBILITY STRENGTH", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)

        self.alpha_label = tk.Label(alpha_header, text=f"{self.alpha.get():.4f}", bg=COLORS['bg_medium'],
                                   fg=COLORS['accent'], font=('Segoe UI', 10, 'bold'))
        self.alpha_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Info label showing minimum alpha constraint
        constraint_label = tk.Label(self.param_frame, 
                                   text="⚠ Min 0.020 required for reliable extraction with HVS model",
                                   bg=COLORS['bg_medium'],
                                   fg=COLORS['text_dim'], font=('Segoe UI', 7, 'italic'))
        constraint_label.pack(anchor=tk.W, pady=(2, 4))

        # Info label showing PSNR expectation
        self.alpha_info = tk.Label(self.param_frame, text="Expected: 42 dB (excellent - imperceptible)",
                                  bg=COLORS['bg_medium'],
                                  fg=COLORS['success'], font=('Segoe UI', 8, 'italic'))
        self.alpha_info.pack(anchor=tk.W, pady=(0, 8))

        # Alpha slider - dark styled and aligned with valid system range
        self.alpha_scale = tk.Scale(
            self.param_frame, from_=0.020, to=0.055, resolution=0.001,
            variable=self.alpha, orient=tk.HORIZONTAL,
            bg=COLORS['bg_medium'], fg=COLORS['accent'],
            troughcolor=COLORS['bg_light'],
            activebackground=COLORS['success'],
            highlightcolor=COLORS['accent'],
            highlightbackground=COLORS['success'],
            highlightthickness=1, borderwidth=0, relief=tk.FLAT,
            sliderrelief=tk.FLAT, sliderlength=28, showvalue=False,
            command=self._update_alpha_info
        )
        self.alpha_scale.pack(fill=tk.X, pady=(2, 10), padx=5)

        # Preset buttons for quick selection
        preset_frame = tk.Frame(self.param_frame, bg=COLORS['bg_medium'])
        preset_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Button(preset_frame, text="MAX INVISIBLE", bg=COLORS['success'], fg='white',
                 font=('Segoe UI', 8, 'bold'), relief=tk.FLAT, padx=2, pady=4,
                 command=lambda: (self.alpha.set(0.020), self._update_alpha_info(0.020))).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        tk.Button(preset_frame, text="BALANCED", bg=COLORS['warning'], fg='white',
                 font=('Segoe UI', 8, 'bold'), relief=tk.FLAT, padx=2, pady=4,
                 command=lambda: (self.alpha.set(0.035), self._update_alpha_info(0.035))).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        tk.Button(preset_frame, text="MAX ROBUST", bg='#ff6b6b', fg='white',
                 font=('Segoe UI', 8, 'bold'), relief=tk.FLAT, padx=2, pady=4,
                 command=lambda: (self.alpha.set(0.055), self._update_alpha_info(0.055))).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Separator
        tk.Frame(self.param_frame, bg=COLORS['border'], height=1).pack(fill=tk.X, pady=(8, 0))

        # AUTO-TUNE SECTION (PROMINENT)
        autotune_label = tk.Label(self.param_frame, text="AUTO-TUNE (BALANCED)", bg=COLORS['bg_medium'],
                fg=COLORS['warning'], font=('Segoe UI', 9, 'bold'))
        autotune_label.pack(anchor=tk.W, pady=(12, 6))

        # Auto-Alpha checkbox (more visible)
        auto_alpha_check = tk.Checkbutton(self.param_frame, text="[OPTIONAL] Auto-Tune Alpha (robustness-aware)",
                  variable=self.auto_alpha_enabled,
                  bg=COLORS['bg_medium'], fg=COLORS['warning'], selectcolor=COLORS['bg_dark'],
                  activebackground=COLORS['bg_medium'], activeforeground=COLORS['warning'],
                  font=('Segoe UI', 9, 'bold'))
        auto_alpha_check.pack(anchor=tk.W, pady=(0, 8))

        auto_alpha_btn = tk.Button(self.param_frame, text="Tune Now",
                                  command=self.auto_tune_alpha,
                                  bg=COLORS['warning'], fg='white',
                                  font=('Segoe UI', 9, 'bold'),
                                  relief=tk.FLAT, padx=8, pady=6)
        auto_alpha_btn.pack(fill=tk.X, pady=(0, 8))

        # Seed
        seed_row = tk.Frame(self.param_frame, bg=COLORS['bg_medium'])
        seed_row.pack(fill=tk.X, pady=(8, 0))

        tk.Label(seed_row, text="Seed ID", bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Segoe UI', 9)).pack(side=tk.LEFT)

        tk.Entry(seed_row, textvariable=self.seed, width=12, bg=COLORS['bg_dark'],
                fg=COLORS['text'], insertbackground=COLORS['text'], relief=tk.FLAT,
                font=('Segoe UI', 9), justify=tk.RIGHT).pack(side=tk.RIGHT, ipady=4)

        # Separator
        tk.Frame(self.param_frame, bg=COLORS['border'], height=1).pack(fill=tk.X, pady=(8, 0))

        # OPTIONS section
        options_label = tk.Label(self.param_frame, text="ADVANCED OPTIONS", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold'))
        options_label.pack(anchor=tk.W, pady=(12, 6))

        # HVS checkbox
        tk.Checkbutton(self.param_frame, text="HVS Masking (smooth area hiding)", variable=self.use_hvs,
                      bg=COLORS['bg_medium'], fg=COLORS['text'], selectcolor=COLORS['bg_dark'],
                      activebackground=COLORS['bg_medium'], activeforeground=COLORS['text'],
                      font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(2, 0))

        # Semantic checkbox - update estimate when toggled
        tk.Checkbutton(self.param_frame, text="Semantic Layer", variable=self.use_semantic,
                      bg=COLORS['bg_medium'], fg=COLORS['text'], selectcolor=COLORS['bg_dark'],
                      activebackground=COLORS['bg_medium'], activeforeground=COLORS['text'],
                      command=lambda: self._update_alpha_info(self.alpha.get()),
                      font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(2, 0))

        # Adversarial checkbox - update estimate when toggled
        tk.Checkbutton(self.param_frame, text="Adversarial Poison (anti-AI)", variable=self.use_poison,
                      bg=COLORS['bg_medium'], fg=COLORS['text'], selectcolor=COLORS['bg_dark'],
                      activebackground=COLORS['bg_medium'], activeforeground=COLORS['text'],
                      command=lambda: self._update_alpha_info(self.alpha.get()),
                      font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(2, 0))

        # Spacer
        tk.Frame(scrollable_frame, bg=COLORS['bg_medium']).pack(fill=tk.BOTH, expand=True)
        self._update_alpha_info(self.alpha.get())
    
    def create_preview(self, parent):
        """Create preview panel with center images and right-side X-ray."""
        # Main container - horizontal layout
        main_container = tk.Frame(parent, bg=COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # LEFT SIDE: Results panel
        results = tk.Frame(main_container, bg=COLORS['bg_medium'], width=280)
        results.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        results.pack_propagate(False)
        
        tk.Label(results, text="RESULTS", bg=COLORS['bg_medium'], 
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, padx=15, pady=(12, 8))
        
        self.result_text = tk.Text(results, bg=COLORS['bg_dark'], 
                                  fg=COLORS['success'], font=('Consolas', 10),
                                  relief=tk.FLAT, padx=15, pady=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.result_text.insert('1.0', "Ready to process images.\n\n• Select an input image\n• Adjust parameters\n• Click EMBED or EXTRACT")
        self.result_text.configure(state='disabled')
        
        # CENTER: Input and Output images (larger, clickable)
        center_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Input/Output row centered
        cards = tk.Frame(center_frame, bg=COLORS['bg_dark'])
        cards.pack(expand=True)
        
        self.input_card = ImageCard(cards, "INPUT", size=350)
        self.input_card.pack(side=tk.LEFT, padx=(0, 15))
        self.input_card.canvas.config(cursor="hand2")
        self.input_card.canvas.bind("<Button-1>", lambda e: self.show_enlarged_image('input'))
        
        arrow = tk.Label(cards, text="→", bg=COLORS['bg_dark'], fg=COLORS['accent'],
                        font=('Segoe UI', 32, 'bold'))
        arrow.pack(side=tk.LEFT, padx=20)
        
        self.output_card = ImageCard(cards, "OUTPUT", size=350)
        self.output_card.pack(side=tk.LEFT, padx=(15, 0))
        self.output_card.canvas.config(cursor="hand2")
        self.output_card.canvas.bind("<Button-1>", lambda e: self.show_enlarged_image('output'))
        
        # RIGHT SIDE: X-Ray visualization (vertical stack)
        xray_frame = tk.Frame(main_container, bg=COLORS['bg_medium'], width=170)
        xray_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        xray_frame.pack_propagate(False)
        
        tk.Label(xray_frame, text="X-RAY", bg=COLORS['bg_medium'], 
                fg=COLORS['text_dim'], font=('Segoe UI', 9, 'bold')).pack(pady=(10, 5))
        
        tk.Label(xray_frame, text="Click to enlarge", bg=COLORS['bg_medium'], 
                fg=COLORS['text_dim'], font=('Segoe UI', 8)).pack(pady=(0, 8))
        
        self.xray_combined_card = ImageCard(xray_frame, "Combined ", size=120)
        self.xray_combined_card.pack(pady=(0, 5))
        self.xray_combined_card.canvas.config(cursor="hand2")
        self.xray_combined_card.canvas.bind("<Button-1>", lambda e: self.show_enlarged_xray('combined'))
        
        self.xray_semantic_card = ImageCard(xray_frame, "Semantic ", size=120)
        self.xray_semantic_card.pack(pady=5)
        self.xray_semantic_card.canvas.config(cursor="hand2")
        self.xray_semantic_card.canvas.bind("<Button-1>", lambda e: self.show_enlarged_xray('semantic'))
        
        self.xray_adversarial_card = ImageCard(xray_frame, "Adversarial ", size=120)
        self.xray_adversarial_card.pack(pady=5)
        self.xray_adversarial_card.canvas.config(cursor="hand2")
        self.xray_adversarial_card.canvas.bind("<Button-1>", lambda e: self.show_enlarged_xray('adversarial'))
    
    def show_enlarged_image(self, panel_name):
        """Show enlarged input or output image in popup."""
        if panel_name == 'input':
            path = self.input_path.get()
            title = "Input Image"
        else:
            path = self.output_path.get()
            title = "Output Image"
        
        if not path or not os.path.exists(path):
            return
        
        try:
            img = Image.open(path)
            orig_w, orig_h = img.size
            
            # Scale to fit screen (max 900x700)
            max_w, max_h = 900, 700
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            
            if scale < 1.0:
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create popup
            popup = tk.Toplevel(self.root)
            popup.title(f"{title} ({orig_w}×{orig_h})")
            popup.configure(bg=COLORS['bg_dark'])
            popup.transient(self.root)
            
            # Center on screen
            popup.geometry(f"{new_w + 20}x{new_h + 60}+{(popup.winfo_screenwidth()-new_w)//2}+{(popup.winfo_screenheight()-new_h)//2}")
            
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(popup, image=photo, bg=COLORS['bg_dark'])
            label.image = photo
            label.pack(padx=10, pady=10)
            
            # Info label
            info = tk.Label(popup, text=f"{os.path.basename(path)} | {orig_w}×{orig_h} px", 
                           bg=COLORS['bg_dark'], fg=COLORS['text_dim'], font=('Segoe UI', 10))
            info.pack(pady=(0, 10))
            
            # Close on click or Escape
            popup.bind("<Button-1>", lambda e: popup.destroy())
            popup.bind("<Escape>", lambda e: popup.destroy())
            popup.focus_set()
            
        except Exception as e:
            print(f"Error enlarging image: {e}")

    def _update_alpha_info(self, value):
        """Update alpha label and PSNR expectation based on slider value."""
        alpha = float(value)
        self.alpha_label.config(text=f"{alpha:.4f}")

        # Estimate PSNR based on alpha value - CALIBRATED from actual measurements
        # Valid range: 0.020-0.055 (aligned with system constraints)
        # Alpha 0.020 → 42dB, 0.025 → 38dB, 0.035 → 35dB, 0.055 → 32dB
        if alpha <= 0.020:
            psnr_est = 42
            invisibility = "EXCELLENT - maximum invisibility"
            color = COLORS['success']
        elif alpha <= 0.025:
            psnr_est = 38
            invisibility = "EXCELLENT - imperceptible"
            color = COLORS['success']
        elif alpha <= 0.035:
            psnr_est = 35
            invisibility = "VERY GOOD - nearly invisible"
            color = COLORS['accent']
        elif alpha <= 0.045:
            psnr_est = 33
            invisibility = "GOOD - balanced quality/robustness"
            color = COLORS['warning']
        else:  # 0.045-0.055
            psnr_est = 32
            invisibility = "FAIR - maximum robustness"
            color = '#ff9800'

        # Adjust estimate for enabled features
        penalty = 0
        if hasattr(self, 'use_semantic') and self.use_semantic.get():
            penalty += 0.5
        if hasattr(self, 'use_poison') and self.use_poison.get():
            penalty += 0.3
        
        psnr_est = psnr_est - penalty
        
        info_text = f"Expected: ≈{psnr_est:.0f} dB ({invisibility})"
        self.alpha_info.config(text=info_text, fg=color)

    def set_mode(self, mode):
        self.mode.set(mode)
        # Reset cross-mode preview state so panels always reflect current mode actions.
        self.output_card.clear()
        self.output_path.set("")
        self.xray_full_combined = None
        self.xray_full_semantic = None
        self.xray_full_adversarial = None
        self.xray_combined_card.clear()
        self.xray_semantic_card.clear()
        self.xray_adversarial_card.clear()

        if mode == 'embed':
            self.embed_btn.configure(bg=COLORS['accent'], fg='white')
            self.decode_btn.configure(bg=COLORS['bg_light'], fg=COLORS['text'])
            self.output_frame.pack(fill=tk.X, pady=(12, 0))
            self.param_frame.pack(fill=tk.X, padx=15, pady=8)
            self.process_btn.configure(text="▶  EMBED WATERMARK")
        else:
            self.decode_btn.configure(bg=COLORS['accent'], fg='white')
            self.embed_btn.configure(bg=COLORS['bg_light'], fg=COLORS['text'])
            self.output_frame.pack_forget()
            self.param_frame.pack_forget()
            self.process_btn.configure(text="[EXTRACT] WATERMARK")
            # Force explicit user selection in extract mode (no preselected carry-over image).
            self.input_path.set("")
            self.img_tensor = None
            self.input_card.clear()
    
    def browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All', '*.*')])
        if path:
            self.input_path.set(path)
            self.input_card.set_image(path)
            # Cache tensor for auto-tune alpha
            try:
                self.img_tensor, _ = load_image(path, resize_to=None)
            except Exception:
                self.img_tensor = None
            if self.mode.get() == 'embed':
                self.output_path.set(os.path.splitext(path)[0] + '_watermarked.png')
                if self.models_loaded and self.auto_alpha_enabled.get() and self.img_tensor is not None:
                    self.auto_tune_alpha()
    
    def browse_output(self):
        path = filedialog.asksaveasfilename(
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')], defaultextension='.png')
        if path:
            self.output_path.set(path)
    
    def load_models_async(self):
        self.status_text.set("⏳ Loading AI models...")
        self.progress.start()
        
        def load():
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _ckpt = os.path.join(_proj_root, 'checkpoints', 'checkpoint_hvs_best.pth')
                self.encoder, self.decoder, self.semantic_encoder, self.semantic_decoder, self.poisoner = \
                    load_models(_ckpt, self.device)
                self.models_loaded = True
                gpu = "GPU" if self.device.type == 'cuda' else "CPU"
                self.root.after(0, lambda: self.status_text.set(f"✅ Ready • {gpu}"))
                self.root.after(0, self.progress.stop)
                if self.img_tensor is not None and self.auto_alpha_enabled.get():
                    self.root.after(0, self.auto_tune_alpha)
            except Exception as e:
                self.root.after(0, lambda: self.status_text.set(f"❌ Model load failed"))
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda ex=e: messagebox.showerror("Error", str(ex)))
        
        threading.Thread(target=load, daemon=True).start()
    
    def process_image(self):
        if self.is_processing:
            messagebox.showinfo("Processing", "Please wait for the current job to finish.")
            return
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models loading...")
            return
        if not self.input_path.get():
            messagebox.showwarning("Input", "Select an input image")
            return
        if self.mode.get() == 'embed' and not self.output_path.get():
            messagebox.showwarning("Output", "Specify output path")
            return

        current_mode = self.mode.get()
        current_input_path = self.input_path.get()

        self.is_processing = True
        self.process_btn.configure(state='disabled')
        if current_mode == 'decode':
            self.output_card.clear()
        
        self.progress.start()
        self.status_text.set("⏳ Processing...")
        
        def process():
            try:
                model_size = (MODEL_SIZE, MODEL_SIZE)
                
                if current_mode == 'embed':
                    # Load at FULL resolution (no resize!)
                    img_tensor, orig_size = load_image(current_input_path, resize_to=None)
                    watermark = generate_watermark(seed=self.seed.get())
                    
                    # R1.1: Validate/fix output path for lossless format
                    output_path = self.output_path.get()
                    if RELIABILITY_AVAILABLE:
                        output_path = validate_save_path(output_path, auto_fix=True)
                        if output_path != self.output_path.get():
                            self.root.after(0, lambda p=output_path: self.output_path.set(p))
                    
                    # R1.2: Clamp alpha to ensure watermark survives quantization
                    requested_alpha = self.alpha.get()
                    effective_alpha = requested_alpha
                    auto_alpha_used = False
                    if self.auto_alpha_enabled.get():
                        effective_alpha = get_optimal_alpha(img_tensor.to(self.device), self.device)
                        # In GUI mode, keep auto-tune invisibility-biased.
                        # Robustness-oriented spikes on smooth images can look worse than
                        # "MAX INVISIBLE", so cap auto-tuned alpha at balanced level.
                        effective_alpha = min(effective_alpha, 0.035)
                        auto_alpha_used = True
                        self.root.after(0, lambda a=effective_alpha: self.alpha.set(a))
                        self.root.after(0, lambda a=effective_alpha: self._update_alpha_info(a))
                    alpha_clamped = False
                    if RELIABILITY_AVAILABLE:
                        clamped_alpha = clamp_alpha(effective_alpha)
                        alpha_clamped = (clamped_alpha != effective_alpha)
                        effective_alpha = clamped_alpha
                    
                    # === EMBED WITH X-RAY VISUALIZATION ===
                    watermarked, xray_data = self.embed_with_xray(
                        img_tensor, watermark, alpha=effective_alpha,
                        use_semantic=self.use_semantic.get(), 
                        use_poison=self.use_poison.get(),
                        use_hvs=self.use_hvs.get())
                    
                    psnr = calculate_psnr(img_tensor.to(self.device), watermarked)
                    
                    # Extract at model size for verification
                    wm_small = F.interpolate(watermarked, size=model_size, mode='bilinear', align_corners=False)
                    extracted, _ = extract_watermark(wm_small, self.decoder, 
                                                    self.semantic_decoder, self.device)
                    ber = calculate_ber(extracted, watermark.to(self.device))
                    
                    final_path = save_image(watermarked, output_path)
                    self.root.after(0, lambda: self.output_card.set_image(final_path))
                    
                    # R1.6: Post-embed verification
                    post_verify_status = ""
                    if RELIABILITY_AVAILABLE:
                        def _extract_for_verify(img_float):
                            img_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
                            img_resized = F.interpolate(img_t, size=model_size, mode='bilinear', align_corners=False)
                            bits, _ = extract_watermark(img_resized, self.decoder, self.semantic_decoder, self.device)
                            return bits[0].cpu().numpy()
                        
                        expected_bits_np = watermark[0].cpu().numpy()
                        post_ber, post_conf, passed = post_embed_verify(
                            saved_path=final_path,
                            extract_fn=_extract_for_verify,
                            expected_bits=expected_bits_np,
                            seed=self.seed.get(),
                        )
                        post_verify_status = f"✅ Verified" if passed else f"⚠️ Verify failed (BER={post_ber:.4f})"
                    
                    # === UPDATE X-RAY VISUALIZATIONS ===
                    self.root.after(0, lambda: self.update_xray_displays(xray_data))
                    
                    psnr_status = '✅ invisible' if psnr > 40 else '👍 nearly invisible' if psnr > 35 else '⚠️ visible'
                    hvs_str = "ON" if self.use_hvs.get() else "OFF"
                    
                    # Get owner info if seed was registered
                    owner_info = self.registry.lookup_seed(self.seed.get())
                    result = f"✅ WATERMARK EMBEDDED (HVS: {hvs_str})\n\n"
                    
                    # Show alpha feedback
                    if auto_alpha_used:
                        result += f"⚙ Auto-tuned α: {requested_alpha:.4f} → {effective_alpha:.4f}\n"
                    elif alpha_clamped:
                        result += f"⚙ Clamped α: {requested_alpha:.4f} → {effective_alpha:.4f}\n"
                    
                    if owner_info:
                        # Increment image count for this owner
                        self.registry.increment_image_count(self.seed.get())
                        result += f"👤 OWNER: {owner_info['owner_name']}\n"
                        result += f"📧 EMAIL: {owner_info['owner_email']}\n"
                        if owner_info.get('organization'):
                            result += f"🏢 ORG:   {owner_info['organization']}\n"
                        if owner_info.get('computer_name'):
                            result += f"💻 REG @: {owner_info['computer_name']}\n"
                        if owner_info.get('location'):
                            result += f"📍 FROM:  {owner_info['location']}\n"
                        result += f"{'─' * 35}\n"
                    
                    result += f"{'─' * 35}\n"
                    result += f"📁 Output:  {os.path.basename(final_path)}\n"
                    result += f"📐 Size:    {orig_size[0]} × {orig_size[1]} px\n"
                    result += f"🔑 Seed:    {self.seed.get()}\n"
                    result += f"💪 Alpha:   {effective_alpha:.4f}"
                    if auto_alpha_used:
                        result += " (adaptive)"
                    if alpha_clamped:
                        result += f" (clamped from {requested_alpha:.4f})"
                    result += "\n"
                    result += f"{'─' * 35}\n"
                    result += f"[PSNR]:    {psnr:.2f} dB {psnr_status}\n"
                    result += f"[BER]:     {ber:.4f} {'[OK]' if ber < 0.1 else '[WARN]'}\n"
                    if post_verify_status:
                        result += f"🔒 R-Trip:  {post_verify_status}\n"
                    
                    self.root.after(0, lambda: self.status_text.set("✅ Done!"))
                
                else:  # decode
                    # Load at full resolution first, then resize with F.interpolate
                    # to match embedding self-check interpolation method
                    img_tensor_full, orig_size = load_image(current_input_path, resize_to=None)
                    img_tensor = F.interpolate(img_tensor_full.to(self.device), size=model_size, 
                                               mode='bilinear', align_corners=False)
                    extracted_bits, logits = extract_watermark(
                        img_tensor, self.decoder, self.semantic_decoder, self.device)
                    
                    expected = generate_watermark(seed=self.seed.get()).to(self.device)
                    ber = calculate_ber(extracted_bits, expected)
                    
                    # Confidence based on BER (0% BER = 100% confidence, 50% BER = 0% confidence)
                    confidence = max(0, (0.5 - ber) / 0.5) * 100
                    
                    bits = ''.join(map(str, extracted_bits[0, :32].int().tolist()))
                    
                    if ber < 0.1:
                        status = "✅ WATERMARK VERIFIED"
                    elif ber < 0.25:
                        status = "⚠️ PARTIAL MATCH"
                    else:
                        status = "❌ NO MATCH"
                    
                    result = f"{status}\n\n"
                    
                    # Lookup owner in registry
                    owner_info = self.registry.lookup_seed(self.seed.get())
                    if owner_info:
                        result += f"👤 OWNER:   {owner_info['owner_name']}\n"
                        result += f"📧 EMAIL:   {owner_info['owner_email']}\n"
                        if owner_info.get('organization'):
                            result += f"🏢 ORG:     {owner_info['organization']}\n"
                        result += f"📅 REG:     {owner_info['created_at']}\n"
                        if owner_info.get('computer_name'):
                            result += f"💻 DEVICE:  {owner_info['computer_name']}\n"
                        if owner_info.get('location'):
                            result += f"📍 ORIGIN:  {owner_info['location']}\n"
                        if owner_info.get('license_type'):
                            result += f"📜 LICENSE: {owner_info['license_type']}\n"
                        result += f"{'─' * 35}\n"
                    
                    result += f"{'─' * 35}\n"
                    result += f"[FILE]:   {os.path.basename(current_input_path)}\n"
                    result += f"📐 Size:    {orig_size[0]} × {orig_size[1]} px\n"
                    result += f"🔑 Seed:    {self.seed.get()}\n"
                    result += f"{'─' * 35}\n"
                    result += f"[BER]:     {ber:.4f}\n"
                    result += f"📈 Conf:    {confidence:.1f}%\n"
                    result += f"{'─' * 35}\n"
                    result += f"🔢 Bits:\n{bits}"
                    self.root.after(0, lambda p=current_input_path: self.output_card.set_image(p))
                    
                    self.root.after(0, lambda: self.status_text.set(f"BER: {ber:.4f}"))
                
                self.root.after(0, lambda: self.update_results(result))
                self.root.after(0, self.progress.stop)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = str(e)  # Capture error message before lambda
                self.root.after(0, lambda: self.status_text.set("❌ Error"))
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
            finally:
                self.root.after(0, lambda: self.process_btn.configure(state='normal'))
                self.root.after(0, lambda: setattr(self, 'is_processing', False))
        
        threading.Thread(target=process, daemon=True).start()
    
    def auto_tune_alpha(self):
        """Auto-calculate optimal alpha based on loaded image texture."""
        if self.img_tensor is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        if self.device is None:
            messagebox.showwarning("Wait", "Models are still loading")
            return
        
        try:
            self.status_text.set("🔄 Analyzing texture...")
            self.progress.start()
            
            def tune():
                try:
                    optimal_alpha = get_optimal_alpha(self.img_tensor.to(self.device), self.device)
                    # Keep GUI auto-tune aligned with invisibility-first expectation.
                    optimal_alpha = min(optimal_alpha, 0.035)
                    
                    # Show feedback if alpha was adjusted
                    user_alpha = self.alpha.get()
                    feedback_msg = f"✨ Alpha tuned: {optimal_alpha:.4f}"
                    if abs(optimal_alpha - user_alpha) > 0.001:
                        feedback_msg += f" (was {user_alpha:.4f})"
                    
                    self.root.after(0, lambda a=optimal_alpha: self.alpha.set(a))
                    self.root.after(0, lambda a=optimal_alpha: self._update_alpha_info(a))
                    self.root.after(0, lambda msg=feedback_msg: self.status_text.set(msg))
                    self.root.after(0, self.progress.stop)
                except Exception as e:
                    self.root.after(0, lambda ex=e: messagebox.showerror("Error", f"Auto-tune failed: {ex}"))
                    self.root.after(0, self.progress.stop)
            
            threading.Thread(target=tune, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-tune alpha: {e}")
    def embed_with_xray(self, img_tensor, watermark, alpha=0.1, use_semantic=True, 
                        use_poison=False, use_hvs=True):
        """Embed watermark and return X-ray visualization data.
        
        Returns:
            watermarked: Final watermarked image tensor
            xray_data: Dict with 'combined', 'semantic', 'adversarial' delta visualizations
        """
        _, C, H, W = img_tensor.shape
        img = img_tensor.to(self.device)
        wm = watermark.to(self.device)
        tile_size = MODEL_SIZE
        
        # Resize to model size
        img_small = F.interpolate(img, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
        
        xray_data = {
            'combined': None,
            'semantic': None, 
            'adversarial': None
        }
        
        with torch.no_grad():
            # Step 1: Base encoder watermark
            encoded_small = self.encoder(img_small, wm, alpha=alpha)
            delta_base = encoded_small - img_small
            
            # Step 2: Semantic watermark (if enabled)
            delta_semantic = torch.zeros_like(delta_base)
            if use_semantic and self.semantic_encoder is not None:
                sem_out = self.semantic_encoder(encoded_small, wm)
                semantic_result = sem_out["protected_images"]
                delta_semantic = semantic_result - encoded_small
                encoded_small = semantic_result
        
        # Step 3: Adversarial poison — MUST run outside no_grad (needs autograd for FGSM/PGD)
        delta_adversarial = torch.zeros_like(delta_base)
        if self.poisoner is not None:
            try:
                poisoned, perturbation = self.poisoner(encoded_small.detach())
                delta_adversarial = perturbation.detach()
                if use_poison:
                    encoded_small = poisoned.detach()
            except Exception as e:
                print(f"Adversarial poisoner warning: {e}")
        
        with torch.no_grad():
            # Combined delta at small scale
            delta_small = encoded_small - img_small
            
            # Upscale all deltas to full resolution with better interpolation
            delta_full = F.interpolate(delta_small, size=(H, W), mode='bicubic', align_corners=False)
            delta_semantic_full = F.interpolate(delta_semantic, size=(H, W), mode='bicubic', align_corners=False)
            delta_adversarial_full = F.interpolate(delta_adversarial, size=(H, W), mode='bicubic', align_corners=False)
            
            if use_hvs:
                # HVS-aware masking with enhanced JND (conservative floor for production)
                perceptual_mask = compute_perceptual_mask_v2(img, self.device, mask_floor=0.65)
                channel_weights = compute_channel_weights(self.device)
                delta_masked = delta_full * perceptual_mask * channel_weights
            else:
                delta_masked = delta_full
            
            # Final watermarked image
            watermarked = (img + delta_masked).clamp(0, 1)
            
            # === Create X-Ray visualizations (auto-enhanced per panel) ===
            
            # Combined delta visualization
            combined_vis = self.delta_to_xray(delta_masked)
            xray_data['combined'] = combined_vis
            
            # Semantic delta visualization (auto-normalized to its own range)
            if use_semantic:
                semantic_vis = self.delta_to_xray(delta_semantic_full)
                xray_data['semantic'] = semantic_vis
            
            # Adversarial delta visualization (always show for X-ray even if not applied)
            if delta_adversarial.abs().sum() > 0:
                adversarial_vis = self.delta_to_xray(delta_adversarial_full)
                xray_data['adversarial'] = adversarial_vis
        
        return watermarked, xray_data
    
    def delta_to_xray(self, delta):
        """Convert delta tensor to X-ray visualization (numpy array).
        
        Uses adaptive auto-normalization so each panel scales to its own
        dynamic range.  This ensures tiny semantic deltas (~1e-4) are just
        as visible as larger adversarial deltas (~1e-2).
        
        Color coding:
        - Red/Yellow: Positive changes (brighter)
        - Blue/Cyan:  Negative changes (darker)
        - Gray:       Neutral (little/no change)
        """
        # delta: [1, 3, H, W]
        delta_np = delta[0].cpu().numpy()  # [3, H, W]
        
        # Average across channels for magnitude and sign direction
        delta_mag = np.mean(np.abs(delta_np), axis=0)   # [H, W]
        delta_sign = np.mean(delta_np, axis=0)           # [H, W]
        
        # --- Adaptive normalization: scale to the panel's own max ---
        mag_max = np.max(delta_mag)
        if mag_max > 1e-10:
            # Normalize magnitude to [0, 1] using percentile for robustness
            p99 = np.percentile(delta_mag, 99.5)
            if p99 > 1e-10:
                delta_mag = np.clip(delta_mag / p99, 0, 1)
            else:
                delta_mag = np.clip(delta_mag / mag_max, 0, 1)
            # Normalize sign proportionally
            sign_max = np.max(np.abs(delta_sign))
            if sign_max > 1e-10:
                delta_sign = delta_sign / sign_max  # [-1, 1]
        else:
            # Essentially zero delta — leave as blank
            delta_mag[:] = 0
            delta_sign[:] = 0
        
        # Create colormap: gray base with red (positive) / blue (negative) overlay
        H, W = delta_mag.shape
        xray = np.ones((H, W, 3), dtype=np.float32) * 0.1  # Dark background
        
        # Positive changes -> warm colors (red/yellow)
        pos_mask = delta_sign > 0
        xray[pos_mask, 0] = 0.1 + delta_mag[pos_mask] * 0.9  # Red
        xray[pos_mask, 1] = 0.1 + delta_mag[pos_mask] * 0.5  # Some green for yellow tint
        xray[pos_mask, 2] = 0.1  # Low blue
        
        # Negative changes -> cool colors (blue/cyan)
        neg_mask = delta_sign < 0
        xray[neg_mask, 0] = 0.1  # Low red
        xray[neg_mask, 1] = 0.1 + delta_mag[neg_mask] * 0.5  # Some green for cyan
        xray[neg_mask, 2] = 0.1 + delta_mag[neg_mask] * 0.9  # Blue
        
        return (xray * 255).astype(np.uint8)
    
    def update_xray_displays(self, xray_data):
        """Update X-ray visualization cards with computed data and store full-res arrays."""
        # Store full-resolution arrays for click-to-enlarge
        self.xray_full_combined = xray_data.get('combined')
        self.xray_full_semantic = xray_data.get('semantic')
        self.xray_full_adversarial = xray_data.get('adversarial')
        
        if xray_data['combined'] is not None:
            self.xray_combined_card.set_image_array(xray_data['combined'], "Combined Δ")
        else:
            self.xray_combined_card.clear()
        
        if xray_data['semantic'] is not None:
            self.xray_semantic_card.set_image_array(xray_data['semantic'], "Semantic Δ")
        else:
            self.xray_semantic_card.clear()
        
        if xray_data['adversarial'] is not None:
            self.xray_adversarial_card.set_image_array(xray_data['adversarial'], "Adversarial Δ")
        else:
            self.xray_adversarial_card.clear()
    
    def show_enlarged_xray(self, panel_name):
        """Open a popup window showing the enlarged X-ray image."""
        titles = {'combined': '[COMBINED]', 'semantic': '[SEMANTIC]', 'adversarial': '[ADVERSARIAL]'}
        arrays = {
            'combined': self.xray_full_combined,
            'semantic': self.xray_full_semantic,
            'adversarial': self.xray_full_adversarial,
        }
        arr = arrays.get(panel_name)
        if arr is None:
            return
        
        popup = tk.Toplevel(self.root)
        popup.title(f"X-Ray — {titles.get(panel_name, panel_name)}")
        popup.configure(bg=COLORS['bg_dark'])
        popup.resizable(False, False)
        
        # Title label
        tk.Label(popup, text=titles.get(panel_name, panel_name),
                bg=COLORS['bg_dark'], fg=COLORS['text'],
                font=('Segoe UI', 14, 'bold')).pack(pady=(15, 8))
        
        # Convert array to PIL image, scale up to ~500px
        if arr.max() <= 1.0:
            img = Image.fromarray((arr * 255).astype(np.uint8))
        else:
            img = Image.fromarray(arr.astype(np.uint8))
        
        # Scale to fit ~500px while preserving aspect ratio
        max_display = 500
        w, h = img.size
        scale = max_display / max(w, h)
        if scale > 1:  # Use NEAREST for upscale to keep pixel detail
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.NEAREST)
        else:
            img.thumbnail((max_display, max_display), Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(popup, image=photo, bg=COLORS['bg_dark'])
        img_label.image = photo  # Keep reference
        img_label.pack(padx=15, pady=(0, 10))
        
        # Hint
        tk.Label(popup, text="Click image or press Esc to close",
                bg=COLORS['bg_dark'], fg=COLORS['text_dim'],
                font=('Segoe UI', 9)).pack(pady=(0, 12))
        
        # Close on click or Escape
        img_label.bind("<Button-1>", lambda e: popup.destroy())
        popup.bind("<Escape>", lambda e: popup.destroy())
        
        # Center on main window
        popup.update_idletasks()
        pw, ph = popup.winfo_width(), popup.winfo_height()
        rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        popup.geometry(f"+{rx + (rw - pw) // 2}+{ry + (rh - ph) // 2}")
        popup.focus_set()
    
    def update_results(self, text):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', text)
        self.result_text.configure(state='disabled')


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    WatermarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
