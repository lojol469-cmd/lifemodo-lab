#!/usr/bin/env python3
"""
🧬 Client Kibalone - Interface Simple
====================================

Client Python pour interagir avec le service Kibalone isolé.

Exemple:
    from kibalone_client import KibaloneClient
    
    client = KibaloneClient()
    
    # Compiler
    result = client.compile('''
        cellule Arbre {
            couleur: "vert"
            age: 3
        }
    ''', target='python')
    
    # Exécuter
    result = client.execute('''
        cellule Test {
            action test() {
                afficher("Hello Kibalone!")
            }
        }
    ''')
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any


class KibaloneClient:
    """Client pour le service Kibalone isolé"""
    
    def __init__(self, service_path: Optional[str] = None):
        """
        Args:
            service_path: Chemin vers kibalone_service.py (auto-détecté si None)
        """
        if service_path is None:
            service_path = Path(__file__).parent / "kibalone_service.py"
        
        self.service_path = str(service_path)
        
        if not os.path.exists(self.service_path):
            raise FileNotFoundError(f"Service non trouvé: {self.service_path}")
    
    def _call_service(self, params: dict, timeout: int = 120) -> Dict[str, Any]:
        """
        Appelle le service avec les paramètres donnés
        
        Args:
            params: Paramètres JSON pour le service
            timeout: Timeout en secondes
            
        Returns:
            Réponse JSON du service
        """
        try:
            result = subprocess.run(
                [sys.executable, self.service_path],
                input=json.dumps(params),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Service échoué (code {result.returncode})",
                    'stderr': result.stderr
                }
            
            # Parser la dernière ligne (JSON)
            try:
                output = json.loads(result.stdout.strip().split('\n')[-1])
                return output
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'Réponse JSON invalide',
                    'raw_output': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timeout après {timeout}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compile(self, code: str, target: str = 'python') -> Dict[str, Any]:
        """
        Compile du code Kibalone
        
        Args:
            code: Code source Kibalone
            target: Cible (python, android, ios, web, desktop)
            
        Returns:
            {
                'success': bool,
                'compiled_code': str (si succès),
                'error': str (si échec)
            }
        """
        return self._call_service({
            'action': 'compile',
            'code': code,
            'target': target
        })
    
    def execute(self, code: str, mode: str = 'simulate') -> Dict[str, Any]:
        """
        Exécute du code Kibalone
        
        Args:
            code: Code source Kibalone
            mode: Mode d'exécution (simulate, deploy)
            
        Returns:
            {
                'success': bool,
                'output': str (si succès),
                'error': str (si échec)
            }
        """
        return self._call_service({
            'action': 'execute',
            'code': code,
            'mode': mode
        }, timeout=180)
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyse du code Kibalone
        
        Args:
            code: Code source Kibalone
            
        Returns:
            {
                'success': bool,
                'analysis': str (si succès),
                'suggestions': list,
                'error': str (si échec)
            }
        """
        return self._call_service({
            'action': 'analyze',
            'code': code
        })
    
    def compile_to_all_targets(self, code: str) -> Dict[str, Dict[str, Any]]:
        """
        Compile vers toutes les cibles supportées
        
        Args:
            code: Code source Kibalone
            
        Returns:
            {
                'python': {...},
                'android': {...},
                'ios': {...},
                'web': {...},
                'desktop': {...}
            }
        """
        targets = ['python', 'android', 'ios', 'web', 'desktop']
        results = {}
        
        for target in targets:
            print(f"🎯 Compilation vers {target}...")
            results[target] = self.compile(code, target)
        
        return results


# =================== Exemples d'utilisation ===================

def example_compile():
    """Exemple: Compiler du code Kibalone"""
    print("=" * 60)
    print("📝 Exemple: Compilation Kibalone")
    print("=" * 60)
    
    client = KibaloneClient()
    
    code = """
cellule Arbre {
    couleur: "vert"
    age: 3
    temperature: 25
    
    action pousser() {
        age = age + 1
        afficher("L'arbre a poussé! Âge: " + age)
    }
    
    action adapter_temperature() {
        si temperature > 30 {
            couleur = "jaune"
            afficher("⚠️ Trop chaud!")
        }
    }
}
"""
    
    result = client.compile(code, target='python')
    
    if result['success']:
        print("✅ Compilation réussie!")
        print("\n📄 Code compilé:")
        print(result['compiled_code'])
    else:
        print(f"❌ Erreur: {result['error']}")
    
    return result


def example_execute():
    """Exemple: Exécuter du code Kibalone"""
    print("\n" + "=" * 60)
    print("▶️  Exemple: Exécution Kibalone")
    print("=" * 60)
    
    client = KibaloneClient()
    
    code = """
cellule TestSimple {
    message: "Hello from Kibalone!"
    
    action demarrer() {
        afficher(message)
        afficher("🧬 Cellule active!")
    }
}

// Activer la cellule
TestSimple.demarrer()
"""
    
    result = client.execute(code, mode='simulate')
    
    if result['success']:
        print("✅ Exécution réussie!")
        print("\n📤 Sortie:")
        print(result['output'])
    else:
        print(f"❌ Erreur: {result['error']}")
    
    return result


def example_analyze():
    """Exemple: Analyser du code Kibalone"""
    print("\n" + "=" * 60)
    print("🔍 Exemple: Analyse Kibalone")
    print("=" * 60)
    
    client = KibaloneClient()
    
    code = """
cellule ComplexeArbre {
    // Code à analyser
    hauteur: 10
    branches: 5
    
    action calculer_surface() {
        // Logique complexe ici
        retourner hauteur * branches * 3.14
    }
}
"""
    
    result = client.analyze(code)
    
    if result['success']:
        print("✅ Analyse réussie!")
        print("\n📊 Résultat:")
        print(result['analysis'])
    else:
        print(f"❌ Erreur: {result['error']}")
    
    return result


def example_multi_target():
    """Exemple: Compiler vers plusieurs cibles"""
    print("\n" + "=" * 60)
    print("🎯 Exemple: Compilation Multi-Cibles")
    print("=" * 60)
    
    client = KibaloneClient()
    
    code = """
cellule Application {
    nom: "MonApp"
    version: "1.0.0"
}
"""
    
    results = client.compile_to_all_targets(code)
    
    print("\n📊 Résultats:")
    for target, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {target.capitalize()}")
    
    return results


if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════╗
║        🧬 Client Kibalone - Exemples d'utilisation       ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Lancer tous les exemples
    example_compile()
    example_execute()
    example_analyze()
    example_multi_target()
    
    print("\n" + "=" * 60)
    print("✨ Tous les exemples terminés!")
    print("=" * 60)
