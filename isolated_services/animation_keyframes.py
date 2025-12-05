"""
Service isolé pour génération d'animations (keyframes)
Utilise le framework isolated_services
"""
import sys
import os

# Ajouter le parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from isolated_services.base import ServiceBase
import torch
import numpy as np
from PIL import Image
import base64
import io
import warnings

# Supprimer warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimationService(ServiceBase):
    """Service pour générer des keyframes d'animation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
    
    def load_model(self):
        """Charger le modèle SD 1.5 depuis cache"""
        if self.pipe is not None:
            return
        
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        
        print("📦 Chargement SD 1.5 depuis cache...", file=sys.stderr)
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Scheduler optimisé
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            beta_schedule="linear"
        )
        
        # Optimisations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
        
        print("✅ Modèle chargé", file=sys.stderr)
    
    def generate_stage_prompt(self, base_prompt, frame_idx, total_frames):
        """Créer prompt pour un stage spécifique"""
        progress = frame_idx / max(1, total_frames - 1)
        
        if frame_idx == 0:
            stage = "beginning pose, initial position, starting stance"
        elif frame_idx == total_frames - 1:
            stage = "final pose, ending position, conclusion stance"
        elif progress < 0.33:
            stage = "early motion, starting movement, initial action"
        elif progress < 0.66:
            stage = "mid action, active movement, dynamic pose"
        else:
            stage = "late motion, concluding movement, finishing action"
        
        quality = "masterpiece, best quality, ultra detailed, sharp focus, crystal clear, professional illustration, perfect composition, clean linework, vibrant colors, consistent character design"
        
        return f"{base_prompt}, {stage}, {quality}"
    
    def process(self, params: dict) -> dict:
        """Générer les keyframes"""
        try:
            # Charger modèle si nécessaire
            self.load_model()
            
            # Extraire params
            prompt = params['prompt']
            num_keyframes = params.get('num_keyframes', 5)
            width = params.get('width', 512)
            height = params.get('height', 512)
            guidance_scale = params.get('guidance_scale', 12.0)
            num_inference_steps = params.get('num_inference_steps', 80)
            seed = params.get('seed', -1)
            
            if seed < 0:
                seed = np.random.randint(0, 2**32)
            
            print(f"🎬 Génération {num_keyframes} keyframes...", file=sys.stderr)
            
            keyframes_b64 = []
            
            for i in range(num_keyframes):
                stage_prompt = self.generate_stage_prompt(prompt, i, num_keyframes)
                negative_prompt = "low quality, blurry, inconsistent style, different character, morphing, deformed"
                
                frame_seed = seed + i * 10
                generator = torch.Generator(device=self.device).manual_seed(frame_seed)
                
                print(f"  Frame {i+1}/{num_keyframes}...", file=sys.stderr)
                
                with torch.no_grad():
                    image = self.pipe(
                        prompt=stage_prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]
                
                # Encoder en base64
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                keyframes_b64.append(img_b64)
            
            print(f"✅ {len(keyframes_b64)} keyframes générées", file=sys.stderr)
            
            return {
                'success': True,
                'keyframes': keyframes_b64,
                'num_keyframes': len(keyframes_b64),
                'seed': seed
            }
            
        except Exception as e:
            import traceback
            print(f"❌ Erreur: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


if __name__ == "__main__":
    service = AnimationService()
    service.run()
