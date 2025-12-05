#!/usr/bin/env python3
"""
🧬 Service Isolé Kibalone
========================

Service qui exécute du code Kibalone dans un processus isolé
sans conflit de dépendances avec le reste de l'application.

Usage:
    echo '{"code": "cellule Test {}", "mode": "compile"}' | python kibalone_service.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from base import ServiceBase
import subprocess
import tempfile
from pathlib import Path


class KibaloneService(ServiceBase):
    """Service pour exécuter du code Kibalone de manière isolée"""
    
    def __init__(self):
        super().__init__()
        self.kibalone_path = os.path.join(
            os.path.dirname(__file__),
            "kibalone-langage"
        )
        self.runner = os.path.join(self.kibalone_path, "run.py")
        
    def compile_kibalone(self, code: str, target: str = "python") -> dict:
        """
        Compile du code Kibalone vers une cible
        
        Args:
            code: Code source Kibalone
            target: Cible de compilation (python, android, ios, web)
            
        Returns:
            dict avec le code compilé ou l'erreur
        """
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kib', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Compiler avec Kibalone
            result = subprocess.run(
                [sys.executable, self.runner, 'compile', temp_file, '--target', target],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.kibalone_path
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'compiled_code': result.stdout,
                    'target': target
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout,
                    'code': result.returncode
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Compilation timeout après 30s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Nettoyer
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def execute_kibalone(self, code: str, mode: str = "simulate") -> dict:
        """
        Exécute du code Kibalone
        
        Args:
            code: Code source Kibalone
            mode: Mode d'exécution (simulate, deploy)
            
        Returns:
            dict avec le résultat de l'exécution
        """
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kib', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Exécuter avec Kibalone
            cmd = [sys.executable, self.runner, 'run', temp_file]
            if mode == "simulate":
                cmd.append('--simulate')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.kibalone_path
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'mode': mode
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout,
                    'code': result.returncode
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Exécution timeout après 60s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Nettoyer
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def analyze_kibalone(self, code: str) -> dict:
        """
        Analyse du code Kibalone (AST, métriques, suggestions)
        
        Args:
            code: Code source Kibalone
            
        Returns:
            dict avec l'analyse du code
        """
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kib', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Analyser avec Kibalone
            result = subprocess.run(
                [sys.executable, self.runner, 'analyze', temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.kibalone_path
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'analysis': result.stdout,
                    'suggestions': []  # Parse du stdout pour extraire suggestions
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Nettoyer
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def process(self, params: dict) -> dict:
        """
        Point d'entrée principal du service
        
        Params attendus:
        {
            "action": "compile" | "execute" | "analyze",
            "code": "code Kibalone",
            "target": "python" (pour compile),
            "mode": "simulate" (pour execute)
        }
        """
        try:
            action = params.get('action', 'compile')
            code = params.get('code', '')
            
            if not code:
                return {
                    'success': False,
                    'error': 'Code Kibalone manquant'
                }
            
            print(f"🧬 Action Kibalone: {action}")
            
            if action == 'compile':
                target = params.get('target', 'python')
                return self.compile_kibalone(code, target)
            
            elif action == 'execute':
                mode = params.get('mode', 'simulate')
                return self.execute_kibalone(code, mode)
            
            elif action == 'analyze':
                return self.analyze_kibalone(code)
            
            else:
                return {
                    'success': False,
                    'error': f'Action inconnue: {action}. Utilisez: compile, execute, analyze'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur interne: {str(e)}'
            }


if __name__ == '__main__':
    service = KibaloneService()
    service.run()
