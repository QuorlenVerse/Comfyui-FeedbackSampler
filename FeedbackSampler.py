import torch
import torch.nn.functional as F
from comfy.samplers import KSampler
import comfy.sample
import comfy.samplers
import comfy.utils
import nodes
import latent_preview
import numpy as np
from PIL import Image
import math

# Try to import scipy for sharpening and noise
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Sharpening and Perlin noise will be disabled. Install with: pip install scipy")


class FeedbackSampler:
    """
    A sampler that feeds finished latent back into itself with zoom functionality.
    Creates deforum-style zooming animations through iterative feedback loops.
    Includes LAB color matching to prevent color bleeding.
    Supports rotation and horizontal panning while ensuring frames are cropped enough to avoid blank areas.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # === Standard KSampler Parameters ===
                "model": ("MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # === Animation Parameters ===
                "zoom_value": ("FLOAT", {"default": 0.05, "min": -0.5, "max": 0.5, "step": 0.0001, "round": 0.0001}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 1000000}),
                "feedback_denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_variation": (["fixed", "increment", "random"], {"default": "increment"}),
                # Rotation & pan
                "rotation_value": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pan_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                
                # === Color & Quality Enhancement ===
                "color_coherence": (["None", "LAB", "RGB", "HSV"], {"default": "LAB"}),
                "noise_amount": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "noise_type": (["gaussian", "perlin"], {"default": "perlin"}),
                "sharpen_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_boost": ("FLOAT", {"default": 0.9, "min": 0.8, "max": 1.5, "step": 0.01}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("final_latent", "all_latents")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"
    
    def match_color_histogram(self, source, reference, mode="LAB"):
        """
        Match color histogram of source image to reference image.
        This is the critical function that prevents color bleeding.
        """
        if mode == "None":
            return source
        
        # Ensure uint8 type
        source = source.astype(np.uint8)
        reference = reference.astype(np.uint8)
        
        if mode == "LAB":
            # Convert to LAB color space (most perceptually uniform)
            source_lab = self.rgb_to_lab(source)
            reference_lab = self.rgb_to_lab(reference)
            
            matched_lab = np.zeros_like(source_lab)
            for i in range(3):
                matched_lab[:, :, i] = self.match_histograms(
                    source_lab[:, :, i], 
                    reference_lab[:, :, i]
                )
            
            result = self.lab_to_rgb(matched_lab)
            
        elif mode == "HSV":
            source_hsv = self.rgb_to_hsv(source)
            reference_hsv = self.rgb_to_hsv(reference)
            
            matched_hsv = np.zeros_like(source_hsv)
            for i in range(3):
                matched_hsv[:, :, i] = self.match_histograms(
                    source_hsv[:, :, i],
                    reference_hsv[:, :, i]
                )
            
            result = self.hsv_to_rgb(matched_hsv)
            
        else:  # RGB
            result = np.zeros_like(source)
            for i in range(3):
                result[:, :, i] = self.match_histograms(
                    source[:, :, i],
                    reference[:, :, i]
                )
        
        return result.astype(np.uint8)
    
    def match_histograms(self, source, reference):
        """
        Match histogram of source channel to reference channel.
        Uses cumulative distribution function (CDF) matching.
        """
        source_values, source_counts = np.unique(source.ravel(), return_counts=True)
        reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)
        
        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]
        
        reference_cdf = np.cumsum(reference_counts).astype(np.float64)
        reference_cdf /= reference_cdf[-1]
        
        interp_values = np.interp(source_cdf, reference_cdf, reference_values)
        
        lookup = np.zeros(256, dtype=reference.dtype)
        for i, val in enumerate(source_values):
            lookup[val] = interp_values[i]
        
        return lookup[source]
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        rgb_norm = rgb.astype(np.float32) / 255.0
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask, 
                              np.power((rgb_norm + 0.055) / 1.055, 2.4),
                              rgb_norm / 12.92)
        xyz = np.zeros_like(rgb_linear)
        xyz[:, :, 0] = rgb_linear[:, :, 0] * 0.4124564 + rgb_linear[:, :, 1] * 0.3575761 + rgb_linear[:, :, 2] * 0.1804375
        xyz[:, :, 1] = rgb_linear[:, :, 0] * 0.2126729 + rgb_linear[:, :, 1] * 0.7151522 + rgb_linear[:, :, 2] * 0.0721750
        xyz[:, :, 2] = rgb_linear[:, :, 0] * 0.0193339 + rgb_linear[:, :, 1] * 0.1191920 + rgb_linear[:, :, 2] * 0.9503041
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))
        lab = np.zeros_like(xyz)
        lab[:, :, 0] = (116 * f[:, :, 1]) - 16
        lab[:, :, 1] = 500 * (f[:, :, 0] - f[:, :, 1])
        lab[:, :, 2] = 200 * (f[:, :, 1] - f[:, :, 2])
        lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0
        lab[:, :, 1] = (lab[:, :, 1] + 128.0)
        lab[:, :, 2] = (lab[:, :, 2] + 128.0)
        return np.clip(lab, 0, 255).astype(np.uint8)
    
    def lab_to_rgb(self, lab):
        """Convert LAB back to RGB with proper bounds checking"""
        lab_float = lab.astype(np.float32)
        lab_float[:, :, 0] = lab_float[:, :, 0] * 100.0 / 255.0
        lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0
        lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0
        fy = (lab_float[:, :, 0] + 16) / 116
        fx = lab_float[:, :, 1] / 500 + fy
        fz = fy - lab_float[:, :, 2] / 200
        fx = np.maximum(fx, 0.0)
        fy = np.maximum(fy, 0.0)
        fz = np.maximum(fz, 0.0)
        mask_x = fx > 0.2068966
        mask_y = fy > 0.2068966
        mask_z = fz > 0.2068966
        xyz = np.zeros_like(lab_float)
        xyz[:, :, 0] = np.where(mask_x, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:, :, 1] = np.where(mask_y, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:, :, 2] = np.where(mask_z, np.power(fz, 3), (fz - 16/116) / 7.787)
        xyz = np.clip(xyz, 0.0, 1.0)
        xyz[:, :, 0] *= 0.95047
        xyz[:, :, 1] *= 1.00000
        xyz[:, :, 2] *= 1.08883
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:, :, 0] = xyz[:, :, 0] *  3.2404542 + xyz[:, :, 1] * -1.5371385 + xyz[:, :, 2] * -0.4985314
        rgb_linear[:, :, 1] = xyz[:, :, 0] * -0.9692660 + xyz[:, :, 1] *  1.8760108 + xyz[:, :, 2] *  0.0415560
        rgb_linear[:, :, 2] = xyz[:, :, 0] *  0.0556434 + xyz[:, :, 1] * -0.2040259 + xyz[:, :, 2] *  1.0572252
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        mask = rgb_linear > 0.0031308
        rgb = np.where(mask,
                      1.055 * np.power(rgb_linear, 1/2.4) - 0.055,
                      12.92 * rgb_linear)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        if np.any(np.isnan(rgb)) or np.any(np.isinf(rgb)):
            print("  WARNING: Invalid values detected in LAB->RGB conversion, using fallback")
            return np.zeros_like(rgb, dtype=np.uint8) + 128
        return rgb
    
    def rgb_to_hsv(self, rgb):
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)
        rc = np.where(deltac != 0, (maxc - r) / deltac, 0)
        gc = np.where(deltac != 0, (maxc - g) / deltac, 0)
        bc = np.where(deltac != 0, (maxc - b) / deltac, 0)
        h = np.zeros_like(r)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        hsv = np.stack([h, s, v], axis=2)
        return (hsv * 255).astype(np.uint8)
    
    def hsv_to_rgb(self, hsv):
        hsv_norm = hsv.astype(np.float32) / 255.0
        h, s, v = hsv_norm[:, :, 0], hsv_norm[:, :, 1], hsv_norm[:, :, 2]
        i = (h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        rgb = np.zeros((*h.shape, 3), dtype=np.float32)
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        mask = (i == 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        return (rgb * 255).astype(np.uint8)
    
    def latent_to_image(self, latent, vae):
        """Convert latent to RGB image for color matching"""
        latent_dict = {"samples": latent}
        decoded = vae.decode(latent_dict["samples"])
        img = decoded[0].cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img
    
    def image_to_latent(self, image, vae):
        """Convert RGB image back to latent"""
        img = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        if hasattr(vae, 'device'):
            img_tensor = img_tensor.to(vae.device)
        elif hasattr(vae, 'first_stage_model'):
            img_tensor = img_tensor.to(next(vae.first_stage_model.parameters()).device)
        latent = vae.encode(img_tensor)
        return latent
    
    def generate_perlin_noise(self, shape, scale=10, octaves=4):
        if not SCIPY_AVAILABLE:
            print("    [Perlin noise unavailable without scipy, using Gaussian]", flush=True)
            return np.random.randn(*shape).astype(np.float32) * 0.5 + 0.5
        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)
        print(f"    [Generating Perlin noise {H}x{W}x{C}...]", end=" ", flush=True)
        for c in range(C):
            channel_noise = np.zeros((H, W), dtype=np.float32)
            for octave in range(octaves):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)
                grid_size = max(4, scale // freq)
                grid_h = H // grid_size + 2
                grid_w = W // grid_size + 2
                grid_noise = np.random.randn(grid_h, grid_w).astype(np.float32) * amp
                upsampled = scipy_zoom(grid_noise, (H / grid_h, W / grid_w), order=1)
                upsampled = upsampled[:H, :W]
                channel_noise += upsampled
            noise[:, :, c] = channel_noise
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        print("Done!", flush=True)
        return noise
    
    def apply_noise_pixel(self, image, amount, noise_type="gaussian"):
        if amount <= 0:
            return image
        img_float = image.astype(np.float32)
        if noise_type == "perlin":
            noise = self.generate_perlin_noise(image.shape, scale=8, octaves=4)
            noise = (noise - 0.5) * 2.0
            noise_scaled = noise * (amount * 30.0)
        else:
            noise = np.random.randn(*image.shape).astype(np.float32)
            noise_scaled = noise * (amount * 15.0)
        noisy = img_float + noise_scaled
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_noise(self, latent, amount):
        if amount <= 0:
            return latent
        noise = torch.randn_like(latent) * amount
        return latent + noise
    
    def apply_sharpening(self, image, amount):
        if amount <= 0 or not SCIPY_AVAILABLE:
            return image
        img_float = image.astype(np.float32)
        blurred = gaussian_filter(img_float, sigma=1.0)
        sharpened = img_float + amount * (img_float - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_contrast(self, image, boost):
        if boost == 1.0:
            return image
        img_float = image.astype(np.float32)
        midpoint = 127.5
        contrasted = (img_float - midpoint) * boost + midpoint
        return np.clip(contrasted, 0, 255).astype(np.uint8)
    
    def transform_latent(self, latent, zoom_factor, rotation_deg, pan_value):
        """
        Apply combined zoom, rotation, and horizontal pan to latent using an affine transform.
        Ensures frames are cropped sufficiently to avoid blank regions by performing a center crop
        before transformation and using border padding when sampling.
        """
        if zoom_factor == 0 and abs(rotation_deg) < 1e-6 and abs(pan_value) < 1e-6:
            return latent
        
        batch, channels, height, width = latent.shape
        # Compute a conservative margin fraction based on requested transforms
        angle_rad = math.radians(abs(rotation_deg))
        rotation_margin = (abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))) * 0.5
        margin_frac = 0.15 + rotation_margin * 0.15 + abs(zoom_factor) * 0.6 + abs(pan_value) * 0.5
        margin_frac = min(0.45, margin_frac)  # clamp to reasonable range
        
        crop_h = max(1, int(height * (1.0 - margin_frac)))
        crop_w = max(1, int(width * (1.0 - margin_frac)))
        
        # Apply pan by shifting the crop center horizontally
        center_x = width // 2 + int(pan_value * width * 0.25)
        center_y = height // 2
        left = int(center_x - crop_w // 2)
        top = int(center_y - crop_h // 2)
        left = max(0, min(width - crop_w, left))
        top = max(0, min(height - crop_h, top))
        
        cropped = latent[:, :, top:top+crop_h, left:left+crop_w]
        
        # Prepare affine matrix for rotation and scale (zoom)
        scale = 1.0 + zoom_factor
        theta = torch.zeros((batch, 2, 3), dtype=latent.dtype, device=latent.device)
        cos_t = math.cos(math.radians(rotation_deg)) * scale
        sin_t = math.sin(math.radians(rotation_deg)) * scale
        # Fill theta for each batch element
        theta[:, 0, 0] = cos_t
        theta[:, 0, 1] = -sin_t
        theta[:, 1, 0] = sin_t
        theta[:, 1, 1] = cos_t
        # No extra translation here — pan was achieved by shifting crop center
        theta[:, 0, 2] = 0.0
        theta[:, 1, 2] = 0.0
        
        # Sample back to original size using border padding to avoid blank areas
        grid = F.affine_grid(theta, size=(batch, channels, height, width), align_corners=False)
        transformed = F.grid_sample(cropped, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return transformed
    
    def zoom_latent(self, latent, zoom_factor):
        """
        Backwards-compatible zoom function retained for external calls. Uses transform_latent (no rotation/pan).
        """
        return self.transform_latent(latent, zoom_factor, 0.0, 0.0)
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise, zoom_value, iterations, feedback_denoise, seed_variation,
               color_coherence, noise_amount, noise_type="perlin", sharpen_amount=0.1, contrast_boost=1.0,
               rotation_value=0.0, pan_value=0.0, vae=None):
        """
        Main sampling function with feedback loop, zoom, rotation, pan, and color coherence.
        """
        import random
        
        if color_coherence != "None" and vae is None:
            print("WARNING: Color coherence requested but no VAE provided. Disabling color coherence.")
            color_coherence = "None"
        
        all_latents = []
        color_reference = None
        current_latent = latent_image["samples"].clone()
        latent_format = latent_image.copy()
        
        print(f"FeedbackSampler v1.4.3: Starting iteration 1/{iterations} with denoise={denoise}")
        latent_format["samples"] = current_latent
        result = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_format, denoise=denoise
        )
        
        current_latent = result[0]["samples"]
        all_latents.append(current_latent.clone())
        
        if color_coherence != "None" and vae is not None:
            color_reference = self.latent_to_image(current_latent, vae)
            print(f"FeedbackSampler: Stored Frame 0 as color reference ({color_coherence} mode)")
        
        for i in range(1, iterations):
            if seed_variation == "fixed":
                iteration_seed = seed
            elif seed_variation == "increment":
                iteration_seed = seed + i
            else:
                iteration_seed = random.randint(0, 0xffffffffffffffff)
            
            print(f"FeedbackSampler: Iteration {i+1}/{iterations} | zoom={zoom_value} | rot={rotation_value} | pan={pan_value} | denoise={feedback_denoise} | seed={iteration_seed} | noise={noise_amount}({noise_type}) | sharpen={sharpen_amount}")
            
            # Apply combined transform (zoom, rotation, pan)
            transformed_latent = self.transform_latent(current_latent, zoom_value, rotation_value, pan_value)
            
            # Color coherence and pixel-space enhancements
            if color_coherence != "None" and vae is not None and color_reference is not None:
                try:
                    print(f"  [1/6] Decoding latent to image...", end=" ", flush=True)
                    current_image = self.latent_to_image(transformed_latent, vae)
                    print(f"OK ({current_image.shape})", flush=True)
                    
                    print(f"  [2/6] Matching colors ({color_coherence})...", end=" ", flush=True)
                    matched_image = self.match_color_histogram(current_image, color_reference, color_coherence)
                    print(f"OK", flush=True)
                    
                    if contrast_boost != 1.0:
                        print(f"  [3/6] Applying contrast boost ({contrast_boost})...", end=" ", flush=True)
                        matched_image = self.apply_contrast(matched_image, contrast_boost)
                        print(f"OK", flush=True)
                    
                    if sharpen_amount > 0:
                        print(f"  [4/6] Applying sharpening ({sharpen_amount})...", end=" ", flush=True)
                        matched_image = self.apply_sharpening(matched_image, sharpen_amount)
                        print(f"OK", flush=True)
                    
                    if noise_amount > 0:
                        print(f"  [5/6] Adding {noise_type} noise ({noise_amount})...", flush=True)
                        matched_image = self.apply_noise_pixel(matched_image, noise_amount, noise_type)
                        print(f"  [5/6] Noise added OK", flush=True)
                    
                    print(f"  [6/6] Encoding image to latent...", end=" ", flush=True)
                    matched_latent = self.image_to_latent(matched_image, vae)
                    print(f"OK ({matched_latent.shape})", flush=True)
                    
                    transformed_latent = matched_latent
                    print(f"  ✓ All enhancements applied successfully", flush=True)
                except Exception as e:
                    import traceback
                    print(f"\n  ✗ ERROR in color/enhancement pipeline: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    print(f"  Continuing without color correction for this frame...", flush=True)
            elif noise_amount > 0:
                transformed_latent = self.apply_noise(transformed_latent, noise_amount)
            
            latent_format["samples"] = transformed_latent
            result = nodes.common_ksampler(
                model, iteration_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_format, denoise=feedback_denoise
            )
            
            current_latent = result[0]["samples"]
            all_latents.append(current_latent.clone())
        
        all_latents_stacked = torch.cat(all_latents, dim=0)
        final_output = {"samples": current_latent}
        all_output = {"samples": all_latents_stacked}
        
        return (final_output, all_output)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FeedbackSampler": FeedbackSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeedbackSampler": "Feedback Sampler"
}
