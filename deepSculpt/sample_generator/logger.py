"""
Tree-structured Hierarchical Logging System for DeepSculpt
This module provides a visual, tree-like logging structure with hierarchical indentation
and color-coded status indicators for displaying complex process workflows.

Key features:
- Tree visualization: Renders process hierarchy using ASCII characters
- Status indicators: Color-coded symbols for different message types
- Section management: Tracks nested process sections with proper indentation
- Color formatting: Consistent color scheme for different message types

Dependencies:
- No internal dependencies (standalone module)

Used by:
- utils.py: For debugging and utility function logging
- shapes.py: For detailed shape generation process logging
- visualization.py: For tracking visualization operations
- sculptor.py: For sculpture generation workflow tracking
- collector.py: For dataset collection process logging
- curator.py: For preprocessing workflow logging

TODO:
- Add log file output capabilities
- Implement log level filtering
- Add timestamp prefixing
- Add support for progress bars
- Create a context manager interface for sections
"""

class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

class TreeSymbols:
    BRANCH = "├── "
    LAST = "└── "
    PIPE = "│   "
    SPACE = "    "
    
    # Status indicators
    ACTION = "[*]"
    SUCCESS = "[+]"
    ERROR = "[!]"
    INFO = "[?]"
    WARNING = "[W]"

# Global variables for tracking tree structure
_indent_level = 0
_indent_stack = []  # Stack to track whether each level is the last item

def print_tree(message, symbol=TreeSymbols.ACTION, is_last=False, level=None, status_color=Colors.CYAN):
    """
    Print a message in tree format with the specified indentation level.
    
    Args:
        message (str): The message to print
        symbol (str): The status symbol to use
        is_last (bool): Whether this is the last item in the current branch
        level (int, optional): Specific indentation level to use. Uses global indent_level if None.
        status_color (str): The color to use for the status symbol
    """
    global _indent_level, _indent_stack
    
    if level is not None:
        current_level = level
    else:
        current_level = _indent_level
    
    # Update the indent stack if this is a new level
    while len(_indent_stack) <= current_level:
        _indent_stack.append(False)
    
    # Update the last item status at the current level
    _indent_stack[current_level] = is_last
    
    # Build the prefix based on the indent stack
    prefix = ""
    for i in range(current_level):
        if _indent_stack[i]:
            prefix += TreeSymbols.SPACE
        else:
            prefix += TreeSymbols.PIPE
    
    # Add the appropriate branch symbol
    if is_last:
        prefix += TreeSymbols.LAST
    else:
        prefix += TreeSymbols.BRANCH
    
    # Print the formatted message
    print(f"{prefix}{status_color}{symbol}{Colors.RESET} {message}")

def begin_section(message, symbol=TreeSymbols.ACTION, is_last=False, status_color=Colors.CYAN):
    """
    Begin a new section in the tree structure.
    
    Args:
        message (str): The section title
        symbol (str): The status symbol to use
        is_last (bool): Whether this is the last section at the current level
        status_color (str): The color to use for the status symbol
    """
    global _indent_level
    print_tree(message, symbol, is_last, status_color=status_color)
    _indent_level += 1

def end_section(message=None, symbol=TreeSymbols.SUCCESS, is_last=True, status_color=Colors.GREEN):
    """
    End the current section in the tree structure.
    
    Args:
        message (str, optional): An optional completion message
        symbol (str): The status symbol to use
        is_last (bool): Whether this is the last message in the section
        status_color (str): The color to use for the status symbol
    """
    global _indent_level
    if message is not None:
        print_tree(message, symbol, is_last, status_color=status_color)
    _indent_level = max(0, _indent_level - 1)

def log_action(message, is_last=False):
    """Print an action message (cyan color)."""
    print_tree(message, TreeSymbols.ACTION, is_last, status_color=Colors.CYAN)

def log_success(message, is_last=True):
    """Print a success message (green color)."""
    print_tree(message, TreeSymbols.SUCCESS, is_last, status_color=Colors.GREEN)

def log_error(message, is_last=True):
    """Print an error message (red color)."""
    print_tree(message, TreeSymbols.ERROR, is_last, status_color=Colors.RED)

def log_info(message, is_last=True):
    """Print an informational message (cyan color)."""
    print_tree(message, TreeSymbols.INFO, is_last, status_color=Colors.CYAN)

def log_warning(message, is_last=True):
    """Print a warning message (yellow color)."""
    print_tree(message, TreeSymbols.WARNING, is_last, status_color=Colors.YELLOW)

if __name__ == "__main__":
    """Example usage demonstrating the logger's capabilities."""
    begin_section("Initialize Workflow")
    begin_section("Loading required modules")
    log_info("This step prepares the environment by importing necessary libraries")
    end_section()
    
    begin_section("Setting up configuration")
    log_info("Configuration values are loaded for proper functionality")
    end_section()
    
    log_success("Workflow initialized successfully")
    end_section()
    
    begin_section("Begin Processing")
    begin_section("Executing main task")
    begin_section("Subtask 1")
    log_success("Completed successfully")
    end_section()
    
    log_warning("Warning in subtask 2")
    end_section()
    
    log_success("Processing complete")
    end_section()
    
    log_success("Workflow complete", is_last=True)