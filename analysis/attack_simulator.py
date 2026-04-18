"""
Attack Simulator
================
Test watermark robustness against various attacks the model was trained on.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project
from inference.inference import (
    load_models, load_image, save_image, generate_watermark,
    extract_watermark, calculate_ber, MODEL_SIZE
)
from training.attacks import (
    diff_jpeg, random_noise, random_blur, random_geometry,
    resize_jpeg_resize, simulated_generative_attack
)


# Dark theme colors
COLORS = {
    'bg_dark': '#0d1117',
    'bg_medium': '#161b22',
    'bg_light': '#21262d',
    'accent': '#58a6ff',
    'success': '#3fb950',
    'warning': '#d29922',
    'error': '#f85149',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
}


class AttackSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Attack Simulator")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # State
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = None
        self.decoder = None
        self.semantic_encoder = None
        self.semantic_decoder = None
        self.poisoner = None
        self.models_loaded = False
        
        self.original_tensor = None
        self.attacked_tensor = None
        self.input_path = tk.StringVar()
        self.seed = tk.IntVar(value=42)
        self.attack_type = tk.StringVar(value='jpeg')
        
        # Attack parameters
        self.jpeg_quality = tk.IntVar(value=70)
        self.noise_std = tk.DoubleVar(value=0.03)
        self.blur_sigma = tk.DoubleVar(value=1.0)
        self.geo_rotate = tk.DoubleVar(value=5.0)
        self.geo_scale = tk.DoubleVar(value=0.05)
        self.gen_strength = tk.StringVar(value='medium')
        self.compound_scale = tk.DoubleVar(value=0.7)
        
        self.build_ui()
        self.load_models_async()
    
    def build_ui(self):
        # Main container
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        tk.Label(main, text="🎯 Attack Simulator", bg=COLORS['bg_dark'],
                fg=COLORS['text'], font=('Segoe UI', 18, 'bold')).pack(anchor='w')
        tk.Label(main, text="Test watermark robustness against trained attacks", 
                bg=COLORS['bg_dark'], fg=COLORS['text_dim'], 
                font=('Segoe UI', 10)).pack(anchor='w', pady=(0, 15))
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True)
        
        # Left panel - Controls
        left = tk.Frame(content, bg=COLORS['bg_medium'], width=300)
        left.pack(side='left', fill='y', padx=(0, 15))
        left.pack_propagate(False)
        
        self.build_controls(left)
        
        # Right panel - Images and Results
        right = tk.Frame(content, bg=COLORS['bg_dark'])
        right.pack(side='left', fill='both', expand=True)
        
        self.build_preview(right)
    
    def build_controls(self, parent):
        pad = {'padx': 15, 'pady': 8}
        
        # Input file
        tk.Label(parent, text="Input Image", bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Segoe UI', 11, 'bold')).pack(anchor='w', **pad)
        
        row = tk.Frame(parent, bg=COLORS['bg_medium'])
        row.pack(fill='x', padx=15)
        
        entry = tk.Entry(row, textvariable=self.input_path, bg=COLORS['bg_light'],
                        fg=COLORS['text'], relief='flat', font=('Segoe UI', 9))
        entry.pack(side='left', fill='x', expand=True, ipady=5)
        
        tk.Button(row, text="📁", command=self.browse_input,
                 bg=COLORS['bg_light'], fg=COLORS['text'], relief='flat').pack(side='right', padx=(5,0))
        
        # Seed
        tk.Label(parent, text="Seed", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 10)).pack(anchor='w', padx=15, pady=(10,0))
        tk.Entry(parent, textvariable=self.seed, bg=COLORS['bg_light'],
                fg=COLORS['text'], relief='flat', width=15).pack(anchor='w', padx=15, ipady=4)
        
        # Separator
        tk.Frame(parent, bg=COLORS['bg_light'], height=1).pack(fill='x', pady=15)
        
        # Attack type
        tk.Label(parent, text="Attack Type", bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Segoe UI', 11, 'bold')).pack(anchor='w', **pad)
        
        attacks = [
            ('JPEG Compression', 'jpeg'),
            ('Gaussian Noise', 'noise'),
            ('Gaussian Blur', 'blur'),
            ('Geometric Transform', 'geometry'),
            ('Color Adjustment', 'color'),
            ('Pixel Dropout', 'dropout'),
            ('Generative AI Sim', 'generative'),
            ('Social Media Pipeline', 'compound'),
            ('All Attacks (Random)', 'random'),
        ]
        
        for text, val in attacks:
            tk.Radiobutton(parent, text=text, variable=self.attack_type, value=val,
                          bg=COLORS['bg_medium'], fg=COLORS['text'],
                          selectcolor=COLORS['bg_light'], activebackground=COLORS['bg_medium'],
                          font=('Segoe UI', 9)).pack(anchor='w', padx=15)
        
        # Separator
        tk.Frame(parent, bg=COLORS['bg_light'], height=1).pack(fill='x', pady=15)
        
        # Attack parameters frame
        self.param_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        self.param_frame.pack(fill='x', **pad)
        
        tk.Label(self.param_frame, text="Parameters", bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        # JPEG quality
        self.add_slider(self.param_frame, "JPEG Quality:", self.jpeg_quality, 10, 100)
        # Noise std
        self.add_slider(self.param_frame, "Noise Std:", self.noise_std, 0.01, 0.10, resolution=0.01)
        # Blur sigma
        self.add_slider(self.param_frame, "Blur Sigma:", self.blur_sigma, 0.5, 2.0, resolution=0.1)
        # Geometry
        self.add_slider(self.param_frame, "Rotation (°):", self.geo_rotate, 0, 15)
        
        # Spacer
        tk.Frame(parent, bg=COLORS['bg_medium']).pack(fill='both', expand=True)
        
        # Buttons
        btn_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        btn_frame.pack(fill='x', padx=15, pady=15)
        
        tk.Button(btn_frame, text="⚡ Apply Attack", command=self.apply_attack,
                 bg=COLORS['accent'], fg='white', font=('Segoe UI', 10, 'bold'),
                 relief='flat', padx=20, pady=8).pack(fill='x', pady=(0, 5))
        
        tk.Button(btn_frame, text="💾 Save Attacked", command=self.save_attacked,
                 bg=COLORS['bg_light'], fg=COLORS['text'], font=('Segoe UI', 10),
                 relief='flat', padx=20, pady=8).pack(fill='x')
    
    def add_slider(self, parent, label, var, from_, to_, resolution=1):
        row = tk.Frame(parent, bg=COLORS['bg_medium'])
        row.pack(fill='x', pady=5)
        tk.Label(row, text=label, bg=COLORS['bg_medium'], fg=COLORS['text_dim'],
                font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')
        tk.Scale(row, variable=var, from_=from_, to=to_, orient='horizontal',
                bg=COLORS['bg_medium'], fg=COLORS['text'], troughcolor=COLORS['bg_light'],
                highlightthickness=0, resolution=resolution, length=150).pack(side='left')
    
    def build_preview(self, parent):
        # Images row
        img_frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        img_frame.pack(fill='x')
        
        # Original
        orig_card = tk.Frame(img_frame, bg=COLORS['bg_medium'])
        orig_card.pack(side='left', padx=(0, 10))
        tk.Label(orig_card, text="Original", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 10)).pack(pady=(10,5))
        self.orig_img_frame = tk.Frame(orig_card, bg=COLORS['bg_light'], width=280, height=280)
        self.orig_img_frame.pack(padx=15, pady=(0, 15))
        self.orig_img_frame.pack_propagate(False)
        self.orig_label = tk.Label(self.orig_img_frame, bg=COLORS['bg_light'],
                                   text="No Image", fg=COLORS['text_dim'])
        self.orig_label.pack(expand=True)
        
        # Attacked
        att_card = tk.Frame(img_frame, bg=COLORS['bg_medium'])
        att_card.pack(side='left')
        tk.Label(att_card, text="Attacked", bg=COLORS['bg_medium'],
                fg=COLORS['text_dim'], font=('Segoe UI', 10)).pack(pady=(10,5))
        self.att_img_frame = tk.Frame(att_card, bg=COLORS['bg_light'], width=280, height=280)
        self.att_img_frame.pack(padx=15, pady=(0, 15))
        self.att_img_frame.pack_propagate(False)
        self.att_label = tk.Label(self.att_img_frame, bg=COLORS['bg_light'],
                                  text="No Image", fg=COLORS['text_dim'])
        self.att_label.pack(expand=True)
        
        # Results
        results_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        results_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        tk.Label(results_frame, text="Results", bg=COLORS['bg_medium'],
                fg=COLORS['text'], font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=15, pady=(10,5))
        
        self.result_text = tk.Text(results_frame, bg=COLORS['bg_light'],
                                   fg=COLORS['text'], relief='flat',
                                   font=('Consolas', 11), height=10, padx=15, pady=10)
        self.result_text.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Keep photo references
        self.orig_photo = None
        self.att_photo = None
    
    def browse_input(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if path:
            self.input_path.set(path)
            self.load_image(path)
    
    def load_image(self, path):
        try:
            # Load and display original
            img = Image.open(path)
            img_display = img.copy()
            img_display.thumbnail((270, 270), Image.Resampling.LANCZOS)
            self.orig_photo = ImageTk.PhotoImage(img_display)
            self.orig_label.configure(image=self.orig_photo, text="")
            
            # Load tensor
            self.original_tensor, _ = load_image(path, resize_to=None)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_models_async(self):
        import threading
        def load():
            try:
                checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'checkpoint_hvs_best.pth')
                self.encoder, self.decoder, self.semantic_encoder, \
                self.semantic_decoder, self.poisoner = load_models(checkpoint_path, self.device)
                self.models_loaded = True
            except Exception as e:
                print(f"Model load error: {e}")
        threading.Thread(target=load, daemon=True).start()
    
    def apply_attack(self):
        if self.original_tensor is None:
            messagebox.showwarning("Input", "Load an image first")
            return
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models still loading...")
            return
        
        import threading
        def process():
            try:
                img = self.original_tensor.to(self.device)
                attack = self.attack_type.get()
                
                # Apply attack
                if attack == 'jpeg':
                    attacked = diff_jpeg(img, quality=self.jpeg_quality.get())
                    attack_desc = f"JPEG Q={self.jpeg_quality.get()}"
                
                elif attack == 'noise':
                    std = self.noise_std.get()
                    attacked = torch.clamp(img + torch.randn_like(img) * std, 0, 1)
                    attack_desc = f"Noise σ={std:.3f}"
                
                elif attack == 'blur':
                    attacked = random_blur(img, sigma_range=(self.blur_sigma.get(), self.blur_sigma.get()))
                    attack_desc = f"Blur σ={self.blur_sigma.get():.1f}"
                
                elif attack == 'geometry':
                    attacked = random_geometry(img, max_rotate=self.geo_rotate.get(), 
                                               max_scale=self.geo_scale.get(), max_trans=0.05)
                    attack_desc = f"Geometry rot={self.geo_rotate.get():.0f}°"
                
                elif attack == 'color':
                    alpha = torch.empty(1, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
                    beta = torch.empty(1, 1, 1, 1, device=self.device).uniform_(-0.08, 0.08)
                    attacked = torch.clamp(alpha * img + beta, 0, 1)
                    attack_desc = f"Color α={alpha.item():.2f}, β={beta.item():.3f}"
                
                elif attack == 'dropout':
                    mask = torch.rand_like(img) > 0.12
                    attacked = img * mask.float()
                    attack_desc = "Dropout 12%"
                
                elif attack == 'generative':
                    strength = self.gen_strength.get()
                    attacked = simulated_generative_attack(img, strength=strength)
                    attack_desc = f"Generative AI ({strength})"
                
                elif attack == 'compound':
                    attacked = resize_jpeg_resize(img, scale_range=(0.5, 0.8), quality_range=(55, 75))
                    attack_desc = "Social Media Pipeline"
                
                elif attack == 'random':
                    # Apply random attack
                    choices = ['jpeg', 'noise', 'blur', 'geometry', 'color', 'generative', 'compound']
                    chosen = np.random.choice(choices)
                    if chosen == 'jpeg':
                        q = np.random.randint(50, 85)
                        attacked = diff_jpeg(img, quality=q)
                        attack_desc = f"Random: JPEG Q={q}"
                    elif chosen == 'noise':
                        std = np.random.uniform(0.02, 0.05)
                        attacked = torch.clamp(img + torch.randn_like(img) * std, 0, 1)
                        attack_desc = f"Random: Noise σ={std:.3f}"
                    elif chosen == 'blur':
                        sigma = np.random.uniform(0.8, 1.5)
                        attacked = random_blur(img, sigma_range=(sigma, sigma))
                        attack_desc = f"Random: Blur σ={sigma:.1f}"
                    elif chosen == 'geometry':
                        attacked = random_geometry(img, max_rotate=10, max_scale=0.1, max_trans=0.1)
                        attack_desc = "Random: Geometry"
                    elif chosen == 'color':
                        alpha = torch.empty(1, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
                        beta = torch.empty(1, 1, 1, 1, device=self.device).uniform_(-0.08, 0.08)
                        attacked = torch.clamp(alpha * img + beta, 0, 1)
                        attack_desc = "Random: Color"
                    elif chosen == 'generative':
                        attacked = simulated_generative_attack(img, strength='medium')
                        attack_desc = "Random: Generative"
                    else:
                        attacked = resize_jpeg_resize(img, scale_range=(0.5, 0.8), quality_range=(55, 75))
                        attack_desc = "Random: Compound"
                else:
                    attacked = img
                    attack_desc = "None"
                
                self.attacked_tensor = attacked
                
                # Extract watermark from attacked image
                model_size = (MODEL_SIZE, MODEL_SIZE)
                att_small = F.interpolate(attacked, size=model_size, mode='bilinear', align_corners=False)
                extracted, _ = extract_watermark(att_small, self.decoder, self.semantic_decoder, self.device)
                
                expected = generate_watermark(seed=self.seed.get()).to(self.device)
                ber = calculate_ber(extracted, expected)
                confidence = max(0, (0.5 - ber) / 0.5) * 100
                
                # Calculate PSNR
                mse = F.mse_loss(img, attacked)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10)).item()
                
                # Status
                if ber < 0.1:
                    status = "✅ WATERMARK SURVIVED"
                    status_color = COLORS['success']
                elif ber < 0.25:
                    status = "⚠️ PARTIAL SURVIVAL"
                    status_color = COLORS['warning']
                else:
                    status = "❌ WATERMARK DESTROYED"
                    status_color = COLORS['error']
                
                # Update UI
                self.root.after(0, lambda: self.display_attacked(attacked))
                
                result = f"{status}\n\n"
                result += f"{'─' * 40}\n"
                result += f"🎯 Attack:      {attack_desc}\n"
                result += f"{'─' * 40}\n"
                result += f"📊 BER:         {ber:.4f} ({int(ber*64)}/64 bits wrong)\n"
                result += f"📈 Confidence:  {confidence:.1f}%\n"
                result += f"📉 PSNR:        {psnr:.2f} dB (attack distortion)\n"
                result += f"{'─' * 40}\n"
                
                if ber < 0.1:
                    result += f"💪 The watermark survived this attack!\n"
                elif ber < 0.25:
                    result += f"⚠️ Watermark partially readable.\n"
                else:
                    result += f"💀 Attack destroyed the watermark.\n"
                
                self.root.after(0, lambda: self.update_results(result))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=process, daemon=True).start()
    
    def display_attacked(self, tensor):
        # Convert tensor to PIL and display
        img_np = tensor[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img_display = img.copy()
        img_display.thumbnail((270, 270), Image.Resampling.LANCZOS)
        self.att_photo = ImageTk.PhotoImage(img_display)
        self.att_label.configure(image=self.att_photo, text="")
    
    def update_results(self, text):
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', text)
        self.result_text.configure(state='disabled')
    
    def save_attacked(self):
        if self.attacked_tensor is None:
            messagebox.showwarning("No Image", "Apply an attack first")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            save_image(self.attacked_tensor, path)
            messagebox.showinfo("Saved", f"Saved to {path}")


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    AttackSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
