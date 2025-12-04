#!/usr/bin/env python3
"""
Script pour convertir la landing page HTML en PDF téléchargeable
"""
import subprocess
import sys

def html_to_pdf():
    """Convertit landing_page.html en PDF avec wkhtmltopdf"""
    try:
        # Vérifier si wkhtmltopdf est installé
        subprocess.run(["wkhtmltopdf", "--version"], capture_output=True, check=True)
        
        print("🔄 Conversion HTML → PDF en cours...")
        
        # Conversion
        result = subprocess.run([
            "wkhtmltopdf",
            "--enable-local-file-access",
            "--print-media-type",
            "--no-stop-slow-scripts",
            "--javascript-delay", "2000",
            "landing_page.html",
            "LifeModo_AI_Lab_Presentation.pdf"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PDF créé : LifeModo_AI_Lab_Presentation.pdf")
            return True
        else:
            print(f"❌ Erreur : {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ wkhtmltopdf n'est pas installé.")
        print("\nInstallation :")
        print("  Ubuntu/Debian : sudo apt-get install wkhtmltopdf")
        print("  macOS         : brew install wkhtmltopdf")
        print("  Windows       : https://wkhtmltopdf.org/downloads.html")
        return False

def create_svg_pdf():
    """Alternative : Créer un PDF avec reportlab si wkhtmltopdf n'est pas dispo"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        
        print("🔄 Création PDF avec reportlab...")
        
        c = canvas.Canvas("LifeModo_AI_Lab_Presentation.pdf", pagesize=A4)
        width, height = A4
        
        # Page 1 : Hero avec logo
        # Charger et dessiner le logo SVG
        try:
            drawing = svg2rlg("logo.svg")
            if drawing:
                drawing.scale(0.5, 0.5)
                renderPDF.draw(drawing, c, width/2 - 50, height - 150)
        except Exception as e:
            print(f"⚠️ Logo SVG non chargé : {e}")
        
        # Titre
        c.setFont("Helvetica-Bold", 36)
        c.setFillColor(HexColor("#00d4ff"))
        c.drawCentredString(width/2, height - 200, "LifeModo AI Lab v2.0")
        
        c.setFont("Helvetica", 18)
        c.setFillColor(HexColor("#666666"))
        c.drawCentredString(width/2, height - 230, "Le Premier Laboratoire IA avec Mode Séparé par Document")
        
        # Features
        y = height - 300
        features = [
            "🗂️ Mode Séparé par PDF : Chaque document a son IA dédiée",
            "🧠 LLM Fine-Tuning : LoRA sur Phi-2 avec quantization 4-bit",
            "👁️ Vision YOLO : Extraction et entraînement automatique",
            "📤 Export Multi-Formats : ONNX, CoreML, TorchScript, OpenVINO",
            "🎵 Audio Generation : MusicGen fine-tuning",
            "🔒 100% Local : Privacy totale, gratuit"
        ]
        
        c.setFont("Helvetica", 14)
        c.setFillColor(HexColor("#000000"))
        for feature in features:
            c.drawString(inch, y, feature)
            y -= 30
        
        # Page 2 : Comparaison
        c.showPage()
        c.setFont("Helvetica-Bold", 24)
        c.setFillColor(HexColor("#ff6b6b"))
        c.drawCentredString(width/2, height - 50, "VS Industrie")
        
        # Tableau comparatif
        y = height - 100
        comparisons = [
            ("Setup Time", "5 min ✓", "2-3 heures", "1-2 heures"),
            ("Coût", "Gratuit ✓", "$1-5/heure", "$1-4/heure"),
            ("Mode Séparé", "✓ Unique", "✗", "✗"),
            ("LLM Fine-tuning", "✓ LoRA Phi-2", "✗ Bedrock", "✗ PaLM"),
            ("Privacy", "✓ 100% Local", "⚠️ Cloud", "⚠️ Cloud"),
        ]
        
        c.setFont("Helvetica-Bold", 12)
        headers = ["Feature", "LifeModo", "AWS", "Google"]
        x_positions = [inch, 2.5*inch, 4*inch, 5.5*inch]
        
        for i, header in enumerate(headers):
            c.drawString(x_positions[i], y, header)
        y -= 20
        
        c.setFont("Helvetica", 10)
        for comp in comparisons:
            for i, val in enumerate(comp):
                c.drawString(x_positions[i], y, val)
            y -= 20
        
        # Footer
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#666666"))
        c.drawCentredString(width/2, 50, "Made with 🔥 in Gabon 🇬🇦 • github.com/lojol469-cmd/lifemodo-lab")
        
        c.save()
        print("✅ PDF créé : LifeModo_AI_Lab_Presentation.pdf")
        return True
        
    except ImportError as e:
        print(f"❌ Bibliothèques manquantes : {e}")
        print("\nInstaller : pip install reportlab svglib")
        return False

if __name__ == "__main__":
    print("📄 Génération du PDF de présentation LifeModo AI Lab\n")
    
    # Essayer wkhtmltopdf d'abord
    if not html_to_pdf():
        print("\n🔄 Tentative avec reportlab...")
        create_svg_pdf()
