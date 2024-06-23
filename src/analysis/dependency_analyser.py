# src/analysis/dependency_analyzer.py

import re
import logging
import json
import requests
from typing import List, Dict

class DependencyAnalyzer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.pypi_url = "https://pypi.org/pypi/{}/json"
        self.safety_url = "https://pyup.io/api/v1/safety/"

    def analyze_dependencies(self, filepath: str, content: str) -> Dict:
        """
        Analyzes dependencies in a given file.
        
        :param filepath: Path to the file
        :param content: Content of the file
        :return: A dictionary with dependencies analysis results
        """
        if filepath.endswith('.py'):
            return self._analyze_python_dependencies(content)
        elif filepath.endswith('.json'):
            return self._analyze_json_dependencies(content)
        else:
            return {"dependencies": "Unsupported file type"}

    def _analyze_python_dependencies(self, content: str) -> Dict:
        """
        Analyzes Python dependencies by looking for import statements.
        
        :param content: Content of the Python file
        :return: A dictionary with dependencies and their versions
        """
        dependencies = self._extract_imports(content)
        dependency_details = {dep: self._get_dependency_details(dep) for dep in dependencies}
        
        return {
            "dependencies": dependency_details,
            "dependency_count": len(dependencies)
        }

    def _extract_imports(self, content: str) -> List[str]:
        """
        Extracts import statements from Python code.
        
        :param content: Content of the Python file
        :return: A list of imported module names
        """
        imports = re.findall(r'^(?:import|from)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
        unique_imports = list(set(imports))
        logging.info(f"Extracted imports: {unique_imports}")
        return unique_imports

    def _get_dependency_details(self, package_name: str) -> Dict:
        """
        Retrieves details of a package from PyPI, including the latest version, license, and security vulnerabilities.
        
        :param package_name: Name of the package
        :return: A dictionary with dependency details
        """
        details = {
            "latest_version": self._get_latest_version(package_name),
            "license": self._get_license(package_name),
            "vulnerabilities": self._get_security_vulnerabilities(package_name)
        }
        return details

    def _get_latest_version(self, package_name: str) -> str:
        """
        Retrieves the latest version of a package from PyPI.
        
        :param package_name: Name of the package
        :return: Latest version of the package
        """
        try:
            response = requests.get(self.pypi_url.format(package_name))
            if response.status_code == 200:
                data = response.json()
                latest_version = data['info']['version']
                return latest_version
            else:
                logging.error(f"Failed to fetch version for {package_name}")
                return "Unknown"
        except Exception as e:
            logging.error(f"Error fetching version for {package_name}: {e}")
            return "Unknown"

    def _get_license(self, package_name: str) -> str:
        """
        Retrieves the license of a package from PyPI.
        
        :param package_name: Name of the package
        :return: License of the package
        """
        try:
            response = requests.get(self.pypi_url.format(package_name))
            if response.status_code == 200:
                data = response.json()
                license = data['info'].get('license', 'Unknown')
                return license
            else:
                logging.error(f"Failed to fetch license for {package_name}")
                return "Unknown"
        except Exception as e:
            logging.error(f"Error fetching license for {package_name}: {e}")
            return "Unknown"

    def _get_security_vulnerabilities(self, package_name: str) -> List[Dict]:
        """
        Checks for security vulnerabilities of a package using the Safety API.
        
        :param package_name: Name of the package
        :return: List of security vulnerabilities
        """
        try:
            response = requests.get(f"{self.safety_url}{package_name}/")
            if response.status_code == 200:
                vulnerabilities = response.json().get('vulnerabilities', [])
                return vulnerabilities
            else:
                logging.error(f"Failed to fetch vulnerabilities for {package_name}")
                return []
        except Exception as e:
            logging.error(f"Error fetching vulnerabilities for {package_name}: {e}")
            return []

    def _analyze_json_dependencies(self, content: str) -> Dict:
        """
        Analyzes JSON dependencies by looking for dependencies key in JSON content.
        
        :param content: Content of the JSON file
        :return: A dictionary with dependencies and their versions
        """
        try:
            data = json.loads(content)
            if 'dependencies' in data:
                dependencies = data['dependencies']
                dependency_details = {dep: self._get_dependency_details(dep) for dep in dependencies}
                return {
                    "dependencies": dependency_details,
                    "dependency_count": len(dependencies)
                }
            else:
                return {"dependencies": "No dependencies found"}
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON content: {e}")
            return {"error": "Invalid JSON format"}

# Example usage
if __name__ == "__main__":
    analyzer = DependencyAnalyzer()
    python_content = """
    import os
    import re
    from datetime import datetime
    """
    json_content = '{"dependencies": {"requests": "2.25.1", "numpy": "1.19.5"}}'
    print(analyzer.analyze_dependencies('example.py', python_content))
    print(analyzer.analyze_dependencies('example.json', json_content))
