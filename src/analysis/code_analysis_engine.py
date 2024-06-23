# src/analysis/code_analysis_engine.py

import os
import ast
import logging
import json
import xml.etree.ElementTree as ET
from analysis.llama_api import LLaMAAPI
from analysis.text_analyser import TextAnalyzer
from analysis.security_analyser import SecurityAnalyzer
from analysis.dependency_analyser import DependencyAnalyzer
from analysis.metrics_calculator import MetricsCalculator

class CodeAnalysisEngine:
    def __init__(self, llama_api_key):
        self.llama_api = LLaMAAPI(llama_api_key)
        self.text_analyzer = TextAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def analyze_codebase(self, directory):
        """
        Walks through the directory and analyzes each file found.
        
        :param directory: Path to the codebase directory
        :return: A dictionary with file paths as keys and their analysis metrics as values
        """
        metrics = {}
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                metrics[filepath] = self.analyze_file(filepath, content)
        self._log_metrics(metrics)
        self._generate_report(metrics)
        return metrics

    def analyze_file(self, filepath, content):
        """
        Analyzes a single file.
        
        :param filepath: Path to the file
        :param content: Content of the file
        :return: A dictionary with analysis results
        """
        try:
            if filepath.endswith('.py'):
                return self.analyze_python_file(filepath, content)
            elif filepath.endswith('.json'):
                return self.analyze_json_file(filepath, content)
            elif filepath.endswith('.xml'):
                return self.analyze_xml_file(filepath, content)
            elif filepath.endswith('.md'):
                return self.analyze_markdown_file(filepath, content)
            else:
                return self.analyze_general_file(filepath, content)
        except Exception as e:
            logging.error(f"Error analyzing {filepath}: {e}")
            return {"error": str(e)}

    def analyze_python_file(self, filepath, content):
        """
        Analyzes a Python file.
        
        :param filepath: Path to the Python file
        :param content: Content of the Python file
        :return: A dictionary with analysis results
        """
        try:
            tree = ast.parse(content)
            llama_analysis = self.llama_api.analyze_text(content)
            text_analysis = self.text_analyzer.analyze_text(content)
            ast_analysis = self.analyze_ast(tree)
            security_analysis = self.security_analyzer.analyze_code(content)
            dependency_analysis = self.dependency_analyzer.analyze_dependencies(content)
            metrics = self.metrics_calculator.calculate_metrics(content)
            
            analysis_results = {
                "llama_analysis": llama_analysis,
                "text_analysis": text_analysis,
                "ast_analysis": ast_analysis,
                "security_analysis": security_analysis,
                "dependency_analysis": dependency_analysis,
                "metrics": metrics
            }
            
            # Perform additional actions based on the analysis
            self._perform_actions(filepath, analysis_results)
            
            return analysis_results
        except SyntaxError as e:
            logging.error(f"Syntax error in {filepath}: {e}")
            return {"error": str(e)}

    def analyze_json_file(self, filepath, content):
        """
        Analyzes a JSON file.
        
        :param filepath: Path to the JSON file
        :param content: Content of the JSON file
        :return: A dictionary with analysis results
        """
        try:
            data = json.loads(content)
            llama_analysis = self.llama_api.analyze_text(content)
            text_analysis = self.text_analyzer.analyze_text(content)
            
            analysis_results = {
                "llama_analysis": llama_analysis,
                "text_analysis": text_analysis,
                "json_analysis": "Valid JSON format"
            }
            
            # Perform additional actions based on the analysis
            self._perform_actions(filepath, analysis_results)
            
            return analysis_results
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {filepath}: {e}")
            return {"error": str(e)}

    def analyze_xml_file(self, filepath, content):
        """
        Analyzes an XML file.
        
        :param filepath: Path to the XML file
        :param content: Content of the XML file
        :return: A dictionary with analysis results
        """
        try:
            tree = ET.ElementTree(ET.fromstring(content))
            llama_analysis = self.llama_api.analyze_text(content)
            text_analysis = self.text_analyzer.analyze_text(content)
            
            analysis_results = {
                "llama_analysis": llama_analysis,
                "text_analysis": text_analysis,
                "xml_analysis": "Valid XML format"
            }
            
            # Perform additional actions based on the analysis
            self._perform_actions(filepath, analysis_results)
            
            return analysis_results
        except ET.ParseError as e:
            logging.error(f"XML parse error in {filepath}: {e}")
            return {"error": str(e)}

    def analyze_markdown_file(self, filepath, content):
        """
        Analyzes a Markdown file.
        
        :param filepath: Path to the Markdown file
        :param content: Content of the Markdown file
        :return: A dictionary with analysis results
        """
        llama_analysis = self.llama_api.analyze_text(content)
        text_analysis = self.text_analyzer.analyze_text(content)
        
        analysis_results = {
            "llama_analysis": llama_analysis,
            "text_analysis": text_analysis,
            "markdown_analysis": "Markdown content analyzed"
        }
        
        # Perform additional actions based on the analysis
        self._perform_actions(filepath, analysis_results)
        
        return analysis_results

    def analyze_general_file(self, filepath, content):
        """
        Analyzes a general file (non-Python).
        
        :param filepath: Path to the file
        :param content: Content of the file
        :return: A dictionary with analysis results
        """
        llama_analysis = self.llama_api.analyze_text(content)
        text_analysis = self.text_analyzer.analyze_text(content)
        
        analysis_results = {
            "llama_analysis": llama_analysis,
            "text_analysis": text_analysis
        }
        
        # Perform additional actions based on the analysis
        self._perform_actions(filepath, analysis_results)
        
        return analysis_results

    def analyze_ast(self, tree):
        """
        Analyzes the Abstract Syntax Tree (AST) of the code.
        
        :param tree: AST of the code
        :return: A dictionary with counts of functions, classes, and imports
        """
        num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        num_classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        num_imports = sum(isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom) for node in ast.walk(tree))
        
        return {
            "num_functions": num_functions,
            "num_classes": num_classes,
            "num_imports": num_imports
        }

    def _perform_actions(self, filepath, analysis_results):
        """
        Performs actions based on the analysis results.
        
        :param filepath: Path to the file
        :param analysis_results: Dictionary with analysis results
        """
        logging.info(f"Analysis for {filepath}: {analysis_results}")
        
        # Example action: creating a new file if certain conditions are met
        if 'suggestion' in analysis_results.get("llama_analysis", {}):
            self._create_new_file(filepath, analysis_results)

    def _create_new_file(self, filepath, analysis_results):
        """
        Creates a new file based on the analysis results.
        
        :param filepath: Path to the original file
        :param analysis_results: Dictionary with analysis results
        """
        new_filepath = filepath + "_enhanced"
        with open(new_filepath, 'w') as f:
            f.write(f"# Enhanced content based on analysis of {filepath}\n")
            f.write(f"# Analysis results: {analysis_results}\n")
        logging.info(f"Created new file: {new_filepath}")

    def _log_metrics(self, metrics):
        """
        Logs the collected metrics.
        
        :param metrics: Dictionary with file paths as keys and their analysis metrics as values
        """
        for filepath, analysis_results in metrics.items():
            logging.info(f"Metrics for {filepath}: {analysis_results}")

    def _generate_report(self, metrics):
        """
        Generates a detailed report of the analysis.
        
        :param metrics: Dictionary with file paths as keys and their analysis metrics as values
        """
        report_path = "analysis_report.txt"
        with open(report_path, 'w') as report_file:
            for filepath, analysis_results in metrics.items():
                report_file.write(f"Metrics for {filepath}:\n{analysis_results}\n\n")
        logging.info(f"Generated analysis report: {report_path}")

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)
