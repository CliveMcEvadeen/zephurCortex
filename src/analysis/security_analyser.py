# src/analysis/security_analyzer.py

import os
import re
import ast
import logging
from typing import List, Dict, Any

class SecurityAnalyzer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def analyze_codebase(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """
        Walks through the directory and analyzes each file for security vulnerabilities.
        
        :param directory: Path to the codebase directory
        :return: A dictionary with file paths as keys and their security analysis results as values
        """
        security_metrics = {}
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                security_metrics[filepath] = self.analyze_file(filepath, content)
        self._log_metrics(security_metrics)
        return security_metrics

    def analyze_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Analyzes a single file for security vulnerabilities.
        
        :param filepath: Path to the file
        :param content: Content of the file
        :return: A dictionary with security analysis results
        """
        if filepath.endswith('.py'):
            return self.analyze_python_file(filepath, content)
        else:
            return self.analyze_general_file(filepath, content)

    def analyze_python_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Analyzes a Python file for security vulnerabilities.
        
        :param filepath: Path to the Python file
        :param content: Content of the Python file
        :return: A dictionary with security analysis results
        """
        tree = ast.parse(content)
        security_issues = {
            'static_analysis': self._perform_static_analysis(tree),
            'secrets_detection': self._detect_secrets(content),
            'input_validation': self._check_input_validation(tree),
            'access_control': self._check_access_control(tree),
            'sql_injection': self._detect_sql_injection(content),
            'xss_detection': self._detect_xss(content),
            'command_injection': self._detect_command_injection(content),
            'buffer_overflow': self._detect_buffer_overflow(content),
            'best_practices': self._check_best_practices(tree)
        }
        return security_issues

    def analyze_general_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """
        Analyzes a general file (non-Python) for security vulnerabilities.
        
        :param filepath: Path to the file
        :param content: Content of the file
        :return: A dictionary with security analysis results
        """
        security_issues = {
            'secrets_detection': self._detect_secrets(content),
            'best_practices': self._check_general_best_practices(content)
        }
        return security_issues

    def _perform_static_analysis(self, tree) -> List[str]:
        """
        Performs static analysis to detect common security vulnerabilities.
        
        :param tree: AST of the code
        :return: List of detected issues
        """
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, 'id', None)
                if func_name in ['exec', 'eval']:
                    issues.append(f"Use of {func_name} detected")
        return issues

    def _detect_secrets(self, content: str) -> List[str]:
        """
        Detects hard-coded secrets in the code.
        
        :param content: The code content
        :return: List of detected secrets
        """
        secrets = []
        patterns = [r'AKIA[0-9A-Z]{16}', r'AIza[0-9A-Za-z-_]{35}', r'[0-9a-fA-F]{32}', r'secret', r'password']
        for pattern in patterns:
            if re.search(pattern, content):
                secrets.append(f"Potential secret detected: {pattern}")
        return secrets

    def _check_input_validation(self, tree) -> List[str]:
        """
        Checks for missing or improper input validation.
        
        :param tree: AST of the code
        :return: List of detected issues
        """
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(arg.arg == 'input' for arg in node.args.args):
                    issues.append(f"Function '{node.name}' may lack proper input validation")
        return issues

    def _check_access_control(self, tree) -> List[str]:
        """
        Ensures proper access controls are in place.
        
        :param tree: AST of the code
        :return: List of detected issues
        """
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'auth' not in {name.id for name in ast.walk(node) if isinstance(name, ast.Name)}:
                    issues.append(f"Function '{node.name}' may lack proper access control")
        return issues

    def _detect_sql_injection(self, content: str) -> List[str]:
        """
        Detects potential SQL injection vulnerabilities.
        
        :param content: The code content
        :return: List of detected issues
        """
        issues = []
        patterns = [r"SELECT .* FROM .* WHERE .*='.*'", r"INSERT INTO .* VALUES \(.*\)", r"UPDATE .* SET .* WHERE .*"]
        for pattern in patterns:
            if re.search(pattern, content):
                issues.append(f"Potential SQL injection detected: {pattern}")
        return issues

    def _detect_xss(self, content: str) -> List[str]:
        """
        Detects potential Cross-Site Scripting (XSS) vulnerabilities.
        
        :param content: The code content
        :return: List of detected issues
        """
        issues = []
        patterns = [r"<script>.*</script>", r"onerror=.*"]
        for pattern in patterns:
            if re.search(pattern, content):
                issues.append(f"Potential XSS detected: {pattern}")
        return issues

    def _detect_command_injection(self, content: str) -> List[str]:
        """
        Detects potential command injection vulnerabilities.
        
        :param content: The code content
        :return: List of detected issues
        """
        issues = []
        patterns = [r"os.system\(", r"subprocess.Popen\(", r"subprocess.call\("]
        for pattern in patterns:
            if re.search(pattern, content):
                issues.append(f"Potential command injection detected: {pattern}")
        return issues

    def _detect_buffer_overflow(self, content: str) -> List[str]:
        """
        Detects potential buffer overflow vulnerabilities.
        
        :param content: The code content
        :return: List of detected issues
        """
        issues = []
        # Buffer overflow detection is more relevant to languages like C/C++, but adding basic checks for Python
        if re.search(r"buffer\s*=\s*['\"].*['\"]", content):
            issues.append("Potential buffer overflow detected")
        return issues

    def _check_best_practices(self, tree) -> List[str]:
        """
        Checks for adherence to security best practices in Python code.
        
        :param tree: AST of the code
        :return: List of detected issues
        """
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'os' and any(alias.name == 'system' for alias in node.names):
                issues.append("Avoid using os.system for better security")
        return issues

    def _check_general_best_practices(self, content: str) -> List[str]:
        """
        Checks for adherence to security best practices in general files.
        
        :param content: The file content
        :return: List of detected issues
        """
        issues = []
        if re.search(r"password\s*=\s*['\"].*['\"]", content):
            issues.append("Avoid hard-coding passwords in the code")
        return issues

    def _log_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Logs the collected security metrics.
        
        :param metrics: Dictionary with file paths as keys and their security analysis metrics as values
        """
        for filepath, analysis_results in metrics.items():
            logging.info(f"Security metrics for {filepath}: {analysis_results}")

# usage
if __name__ == "__main__":
    analyzer = SecurityAnalyzer()
    sample_code = """
    import os

    def login(username, password):
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        os.system(f"echo 'Logged in as {username}'")

    """
    security_metrics = analyzer.analyze_file("example.py", sample_code)
    print(security_metrics)
