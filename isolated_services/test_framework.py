#!/usr/bin/env python3
"""
🧪 TEST SUITE POUR ISOLATED SERVICES FRAMEWORK
===============================================
Teste toutes les fonctionnalités du framework.
"""
import sys
import json
import subprocess
import time
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def test_base_service():
    """Test 1: Service de base simple"""
    print(f"\n{Colors.BLUE}Test 1: Service de base{Colors.END}")
    
    # Créer un service minimal
    service_code = '''
from base import ServiceBase

class TestService(ServiceBase):
    def process(self, params):
        return {"result": f"Hello {params.get('name', 'World')}!"}

if __name__ == '__main__':
    TestService().run()
'''
    
    # Écrire dans fichier temporaire
    test_file = Path('/tmp/test_service.py')
    test_file.write_text(service_code)
    
    # Copier base.py
    import shutil
    base_path = Path(__file__).parent / 'base.py'
    shutil.copy(base_path, '/tmp/base.py')
    
    # Appeler service
    params = json.dumps({'name': 'Alice'})
    result = subprocess.run(
        [sys.executable, str(test_file)],
        input=params,
        capture_output=True,
        text=True,
        cwd='/tmp'
    )
    
    # Vérifier
    if result.returncode == 0:
        output = json.loads(result.stdout.strip().split('\n')[-1])
        if output.get('result') == 'Hello Alice!' and output.get('success'):
            print(f"{Colors.GREEN}✅ PASS: Service de base fonctionne{Colors.END}")
            return True
    
    print(f"{Colors.RED}❌ FAIL: {result.stderr}{Colors.END}")
    return False


def test_validation():
    """Test 2: Validation des entrées"""
    print(f"\n{Colors.BLUE}Test 2: Validation entrées{Colors.END}")
    
    service_code = '''
from base import ServiceBase

class ValidatingService(ServiceBase):
    def validate_input(self, params):
        if 'required' not in params:
            raise ValueError("Missing required parameter")
    
    def process(self, params):
        return {"ok": True}

if __name__ == '__main__':
    ValidatingService().run()
'''
    
    test_file = Path('/tmp/validating_service.py')
    test_file.write_text(service_code)
    
    # Test avec params invalides
    params = json.dumps({'invalid': 'data'})
    result = subprocess.run(
        [sys.executable, str(test_file)],
        input=params,
        capture_output=True,
        text=True,
        cwd='/tmp'
    )
    
    # Devrait échouer avec returncode 1
    if result.returncode == 1:
        output = json.loads(result.stdout.strip().split('\n')[-1])
        if not output.get('success') and 'required' in output.get('error', ''):
            print(f"{Colors.GREEN}✅ PASS: Validation fonctionne{Colors.END}")
            return True
    
    print(f"{Colors.RED}❌ FAIL: Validation ne fonctionne pas{Colors.END}")
    return False


def test_error_handling():
    """Test 3: Gestion d'erreurs"""
    print(f"\n{Colors.BLUE}Test 3: Gestion erreurs{Colors.END}")
    
    service_code = '''
from base import ServiceBase

class ErrorService(ServiceBase):
    def process(self, params):
        if params.get('crash'):
            raise RuntimeError("Intentional crash")
        return {"ok": True}

if __name__ == '__main__':
    ErrorService().run()
'''
    
    test_file = Path('/tmp/error_service.py')
    test_file.write_text(service_code)
    
    # Provoquer une erreur
    params = json.dumps({'crash': True})
    result = subprocess.run(
        [sys.executable, str(test_file)],
        input=params,
        capture_output=True,
        text=True,
        cwd='/tmp'
    )
    
    # Vérifier erreur bien formatée
    if result.returncode == 1:
        output = json.loads(result.stdout.strip().split('\n')[-1])
        if (not output.get('success') and 
            'Intentional crash' in output.get('error', '') and
            output.get('error_type') == 'RuntimeError' and
            'traceback' in output):
            print(f"{Colors.GREEN}✅ PASS: Erreurs bien gérées{Colors.END}")
            return True
    
    print(f"{Colors.RED}❌ FAIL: Gestion erreurs défaillante{Colors.END}")
    return False


def test_metadata():
    """Test 4: Métadonnées automatiques"""
    print(f"\n{Colors.BLUE}Test 4: Métadonnées{Colors.END}")
    
    service_code = '''
from base import ServiceBase
import time

class SlowService(ServiceBase):
    def process(self, params):
        time.sleep(0.1)  # Simule traitement
        return {"data": "ok"}

if __name__ == '__main__':
    SlowService().run()
'''
    
    test_file = Path('/tmp/slow_service.py')
    test_file.write_text(service_code)
    
    params = json.dumps({})
    result = subprocess.run(
        [sys.executable, str(test_file)],
        input=params,
        capture_output=True,
        text=True,
        cwd='/tmp'
    )
    
    if result.returncode == 0:
        output = json.loads(result.stdout.strip().split('\n')[-1])
        meta = output.get('_meta', {})
        
        if ('duration_seconds' in meta and 
            'timestamp' in meta and
            meta['duration_seconds'] >= 0.1):
            print(f"{Colors.GREEN}✅ PASS: Métadonnées présentes{Colors.END}")
            print(f"   Duration: {meta['duration_seconds']:.3f}s")
            return True
    
    print(f"{Colors.RED}❌ FAIL: Métadonnées manquantes{Colors.END}")
    return False


def test_client():
    """Test 5: Client helper"""
    print(f"\n{Colors.BLUE}Test 5: Client helper{Colors.END}")
    
    try:
        from client import ServiceClient
        
        # Utiliser service de test existant
        test_file = Path('/tmp/test_service.py')
        if not test_file.exists():
            print(f"{Colors.YELLOW}⚠️  SKIP: Service de test non trouvé{Colors.END}")
            return True
        
        client = ServiceClient(str(test_file))
        result = client.call({'name': 'Bob'}, timeout=5)
        
        if result.get('result') == 'Hello Bob!':
            print(f"{Colors.GREEN}✅ PASS: Client fonctionne{Colors.END}")
            return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ FAIL: {e}{Colors.END}")
        return False
    
    return False


def test_concurrent():
    """Test 6: Appels parallèles"""
    print(f"\n{Colors.BLUE}Test 6: Appels parallèles{Colors.END}")
    
    try:
        import concurrent.futures
        from client import ServiceClient
        
        test_file = Path('/tmp/test_service.py')
        if not test_file.exists():
            print(f"{Colors.YELLOW}⚠️  SKIP: Service de test non trouvé{Colors.END}")
            return True
        
        client = ServiceClient(str(test_file))
        
        # Lancer 5 appels en parallèle
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(client.call, {'name': f'User{i}'})
                for i in range(5)
            ]
            results = [f.result() for f in futures]
        duration = time.time() - start
        
        # Vérifier tous les résultats
        if all(r.get('success') for r in results):
            print(f"{Colors.GREEN}✅ PASS: Parallélisme OK ({duration:.2f}s){Colors.END}")
            return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ FAIL: {e}{Colors.END}")
        return False
    
    return False


def main():
    """Lance tous les tests."""
    print(f"""
{Colors.BLUE}{'='*60}
🧪 TEST SUITE - ISOLATED SERVICES FRAMEWORK
{'='*60}{Colors.END}
    """)
    
    tests = [
        test_base_service,
        test_validation,
        test_error_handling,
        test_metadata,
        test_client,
        test_concurrent,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"{Colors.RED}❌ EXCEPTION: {e}{Colors.END}")
            results.append(False)
    
    # Résumé
    passed = sum(results)
    total = len(results)
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    if passed == total:
        print(f"{Colors.GREEN}✅ ALL TESTS PASSED ({passed}/{total}){Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  SOME TESTS FAILED ({passed}/{total}){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
