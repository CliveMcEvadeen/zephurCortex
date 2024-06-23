import ast
import re
import logging
from collections import defaultdict

class MetricsCalculator:
    
    def __init__(self):

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def calculate_metrics(self, content: str) -> dict:
        """
        Calculate various metrics for the given code content.

        :param content: The code content to analyze.
        :return: A dictionary with various code metrics.
        """
        metrics = {
            'lines_of_code': self._calculate_loc(content),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(content),
            'maintainability_index': self._calculate_maintainability_index(content),
            'code_duplication': self._calculate_code_duplication(content),
            'function_metrics': self._calculate_function_metrics(content),
            'class_metrics': self._calculate_class_metrics(content),
            'documentation_metrics': self._calculate_documentation_metrics(content),
            'dependency_metrics': self._calculate_dependency_metrics(content)
        }
        logging.info(f"Calculated metrics: {metrics}")

        return metrics

    def _calculate_loc(self, content: str) -> dict:
        """
        Calculate lines of code metrics.

        :param content: The code content.
        :return: A dictionary with lines of code metrics.
        """
        lines = content.split('\n')
        total_lines = len(lines)
        blank_lines = len([line for line in lines if not line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])

        return {
            'total_lines': total_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines
        }

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """
        Calculate the cyclomatic complexity of the code.

        :param content: The code content.
        :return: Cyclomatic complexity.
        """
        tree = ast.parse(content)
        complexity = self._get_cyclomatic_complexity(tree)

        return complexity

    def _get_cyclomatic_complexity(self, node) -> int:
        """
        Helper function to recursively calculate cyclomatic complexity.

        :param node: AST node.
        :return: Cyclomatic complexity.
        """
        complexity = 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)):
                complexity += 1
            complexity += self._get_cyclomatic_complexity(child)
        return complexity

    def _calculate_maintainability_index(self, content: str) -> float:
        """
        Calculate the maintainability index of the code.

        :param content: The code content.
        :return: Maintainability index.
        """
        loc_metrics = self._calculate_loc(content)
        complexity = self._calculate_cyclomatic_complexity(content)
        num_comments = loc_metrics['comment_lines']
        loc = loc_metrics['total_lines']

        if loc == 0:
            
            return 100.0

        maintainability_index = (
            max(0, (171 - 5.2 * (loc_metrics['total_lines']) - 0.23 * complexity - 16.2 * num_comments) / 171) * 100
        )
        return maintainability_index

    def _calculate_code_duplication(self, content: str) -> dict:
        """
        Detect code duplication.

        :param content: The code content.
        :return: A dictionary with code duplication metrics.
        """
        lines = content.split('\n')
        duplicates = defaultdict(int)

        for i, line in enumerate(lines):
            if len(line.strip()) > 10:
                duplicates[line.strip()] += 1

        duplicated_lines = {line: count for line, count in duplicates.items() if count > 1}
        duplication_percentage = (sum(duplicated_lines.values()) / len(lines)) * 100

        return {
            'duplicated_lines': duplicated_lines,
            'duplication_percentage': duplication_percentage
        }

    def _calculate_function_metrics(self, content: str) -> dict:
        """
        Calculate function metrics.

        :param content: The code content.
        :return: A dictionary with function metrics.
        """
        tree = ast.parse(content)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        num_functions = len(functions)
        total_function_length = sum(len(node.body) for node in functions)
        average_function_length = total_function_length / num_functions if num_functions > 0 else 0

        return {
            'num_functions': num_functions,
            'total_function_length': total_function_length,
            'average_function_length': average_function_length,
            'function_complexity': self._calculate_cyclomatic_complexity(content)
        }

    def _calculate_class_metrics(self, content: str) -> dict:
        """
        Calculate class metrics.

        :param content: The code content.
        :return: A dictionary with class metrics.
        """
        tree = ast.parse(content)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        num_classes = len(classes)
        total_class_size = sum(len(node.body) for node in classes)
        average_class_size = total_class_size / num_classes if num_classes > 0 else 0

        return {
            'num_classes': num_classes,
            'total_class_size': total_class_size,
            'average_class_size': average_class_size,
            'class_complexity': self._calculate_cyclomatic_complexity(content)
        }

    def _calculate_documentation_metrics(self, content: str) -> dict:
        """
        Calculate documentation metrics.

        :param content: The code content.
        :return: A dictionary with documentation metrics.
        """
        lines = content.split('\n')
        docstring_lines = sum(
            1 for line in lines if line.strip().startswith('"""') or line.strip().startswith("'''")
        )

        return {
            'docstring_lines': docstring_lines,
            'documentation_percentage': (docstring_lines / len(lines)) * 100
        }

    def _calculate_dependency_metrics(self, content: str) -> dict:
        """
        Calculate dependency metrics.

        :param content: The code content.
        :return: A dictionary with dependency metrics.
        """
        imports = re.findall(r'^(?:import|from)\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
        unique_imports = list(set(imports))
        unused_imports = [imp for imp in unique_imports if content.count(imp) == 1]

        return {
            'total_imports': len(unique_imports),
            'unused_imports': unused_imports,
            'used_imports': [imp for imp in unique_imports if imp not in unused_imports]
        }
    

# Example usage
if __name__ == "__main__":
    calculator = MetricsCalculator()
    sample_code = """
    import os
    import re

    def foo():
        pass

    def bar():
        pass
    """
    metrics = calculator.calculate_metrics(sample_code)
    print(metrics)
