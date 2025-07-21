#!/usr/bin/env python3
"""
AI-Assisted Coding Environment
An IDE with advanced AI-powered coding assistance features.
"""

import os
import re
import ast
import openai
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict, Any, Optional, Tuple
import threading
import time

class CodeAnalyzer:
    def __init__(self):
        """Initialize the code analyzer."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze_syntax(self, code: str) -> Dict[str, Any]:
        """
        Analyze code syntax and detect errors.
        
        Args:
            code: Code to analyze
            
        Returns:
            Analysis results
        """
        analysis = {
            "syntax_errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Parse Python code
            ast.parse(code)
            analysis["syntax_valid"] = True
        except SyntaxError as e:
            analysis["syntax_valid"] = False
            analysis["syntax_errors"].append({
                "line": e.lineno,
                "message": str(e),
                "type": "SyntaxError"
            })
        
        # Check for common issues
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for unused imports (simplified)
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                module_name = line.split()[1].split('.')[0]
                if module_name not in code:
                    analysis["warnings"].append({
                        "line": i,
                        "message": f"Potentially unused import: {module_name}",
                        "type": "UnusedImport"
                    })
            
            # Check for long lines
            if len(line) > 100:
                analysis["warnings"].append({
                    "line": i,
                    "message": "Line too long (>100 characters)",
                    "type": "LineTooLong"
                })
        
        return analysis
    
    def suggest_completion(self, code: str, cursor_position: int) -> List[str]:
        """
        Suggest code completions based on context.
        
        Args:
            code: Current code
            cursor_position: Cursor position in the code
            
        Returns:
            List of completion suggestions
        """
        # Get context around cursor
        lines = code[:cursor_position].split('\n')
        current_line = lines[-1] if lines else ""
        
        # Simple completion suggestions
        suggestions = []
        
        # Python keywords and common patterns
        if current_line.strip().endswith('.'):
            suggestions.extend([
                "append(", "extend(", "insert(", "remove(", "pop()",
                "split()", "join()", "replace(", "strip()", "lower()", "upper()"
            ])
        elif current_line.strip().startswith('def '):
            suggestions.extend([
                "def function_name(self):",
                "def __init__(self):",
                "def __str__(self):"
            ])
        elif current_line.strip().startswith('class '):
            suggestions.extend([
                "class ClassName:",
                "class ClassName(object):",
                "class ClassName(Exception):"
            ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """
        Generate code from natural language description.
        
        Args:
            description: Natural language description
            language: Programming language
            
        Returns:
            Generated code
        """
        try:
            prompt = f"""
            Generate {language} code for the following description:
            {description}
            
            Provide clean, well-commented code with proper error handling.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are an expert {language} programmer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"# Error generating code: {str(e)}"
    
    def suggest_fixes(self, code: str, error_message: str) -> str:
        """
        Suggest fixes for code errors.
        
        Args:
            code: Code with errors
            error_message: Error message
            
        Returns:
            Suggested fix
        """
        try:
            prompt = f"""
            The following code has an error:
            
            Code:
            {code}
            
            Error:
            {error_message}
            
            Suggest a fix for this error. Provide the corrected code with explanation.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error suggesting fix: {str(e)}"

class AICodeEditor:
    def __init__(self, root):
        """Initialize the AI code editor."""
        self.root = root
        self.root.title("AI-Assisted Coding Environment")
        self.root.geometry("1200x800")
        
        self.analyzer = CodeAnalyzer()
        self.current_file = None
        
        self.setup_ui()
        self.setup_bindings()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create menu bar
        self.create_menu()
        
        # Create toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar, text="New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Run", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Generate Code", command=self.show_code_generator).pack(side=tk.LEFT, padx=2)
        
        # Create main paned window
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Code editor
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)
        
        # Code editor
        ttk.Label(left_frame, text="Code Editor").pack(anchor=tk.W)
        self.code_editor = scrolledtext.ScrolledText(
            left_frame, 
            wrap=tk.NONE, 
            font=("Consolas", 11),
            undo=True
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Right panel - AI assistance
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # AI assistance notebook
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        ttk.Label(analysis_frame, text="Code Analysis").pack(anchor=tk.W)
        self.analysis_text = scrolledtext.ScrolledText(
            analysis_frame, 
            height=10, 
            font=("Consolas", 9)
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Suggestions tab
        suggestions_frame = ttk.Frame(self.notebook)
        self.notebook.add(suggestions_frame, text="Suggestions")
        
        ttk.Label(suggestions_frame, text="AI Suggestions").pack(anchor=tk.W)
        self.suggestions_listbox = tk.Listbox(suggestions_frame, font=("Consolas", 9))
        self.suggestions_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Output tab
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="Output")
        
        ttk.Label(output_frame, text="Code Output").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            height=10, 
            font=("Consolas", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_command(label="Analyze Code", command=self.analyze_code)
        ai_menu.add_command(label="Generate Code", command=self.show_code_generator)
        ai_menu.add_command(label="Suggest Fixes", command=self.suggest_fixes)
    
    def setup_bindings(self):
        """Set up event bindings."""
        self.code_editor.bind('<KeyRelease>', self.on_code_change)
        self.code_editor.bind('<Button-1>', self.on_cursor_move)
        self.suggestions_listbox.bind('<Double-Button-1>', self.insert_suggestion)
    
    def on_code_change(self, event=None):
        """Handle code changes."""
        # Auto-analyze code after a delay
        self.root.after(1000, self.auto_analyze)
    
    def on_cursor_move(self, event=None):
        """Handle cursor movement."""
        # Update suggestions based on cursor position
        self.update_suggestions()
    
    def auto_analyze(self):
        """Automatically analyze code."""
        code = self.code_editor.get(1.0, tk.END)
        if code.strip():
            threading.Thread(target=self.analyze_code_background, args=(code,), daemon=True).start()
    
    def analyze_code_background(self, code):
        """Analyze code in background thread."""
        analysis = self.analyzer.analyze_syntax(code)
        
        # Update UI in main thread
        self.root.after(0, self.update_analysis_display, analysis)
    
    def update_analysis_display(self, analysis):
        """Update the analysis display."""
        self.analysis_text.delete(1.0, tk.END)
        
        if analysis["syntax_valid"]:
            self.analysis_text.insert(tk.END, "✓ Syntax is valid\n\n")
        else:
            self.analysis_text.insert(tk.END, "✗ Syntax errors found:\n")
            for error in analysis["syntax_errors"]:
                self.analysis_text.insert(tk.END, f"Line {error['line']}: {error['message']}\n")
            self.analysis_text.insert(tk.END, "\n")
        
        if analysis["warnings"]:
            self.analysis_text.insert(tk.END, "Warnings:\n")
            for warning in analysis["warnings"]:
                self.analysis_text.insert(tk.END, f"Line {warning['line']}: {warning['message']}\n")
    
    def update_suggestions(self):
        """Update code suggestions."""
        code = self.code_editor.get(1.0, tk.END)
        cursor_pos = self.code_editor.index(tk.INSERT)
        
        # Convert cursor position to character index
        char_pos = len(self.code_editor.get(1.0, cursor_pos))
        
        suggestions = self.analyzer.suggest_completion(code, char_pos)
        
        self.suggestions_listbox.delete(0, tk.END)
        for suggestion in suggestions:
            self.suggestions_listbox.insert(tk.END, suggestion)
    
    def insert_suggestion(self, event=None):
        """Insert selected suggestion."""
        selection = self.suggestions_listbox.curselection()
        if selection:
            suggestion = self.suggestions_listbox.get(selection[0])
            self.code_editor.insert(tk.INSERT, suggestion)
    
    def new_file(self):
        """Create a new file."""
        self.code_editor.delete(1.0, tk.END)
        self.current_file = None
        self.status_bar.config(text="New file")
    
    def open_file(self):
        """Open a file."""
        filename = filedialog.askopenfilename(
            title="Open File",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.code_editor.delete(1.0, tk.END)
                self.code_editor.insert(1.0, content)
                self.current_file = filename
                self.status_bar.config(text=f"Opened: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    def save_file(self):
        """Save the current file."""
        if self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.save_as_file()
    
    def save_as_file(self):
        """Save file with a new name."""
        filename = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            self.save_to_file(filename)
            self.current_file = filename
    
    def save_to_file(self, filename):
        """Save content to file."""
        try:
            content = self.code_editor.get(1.0, tk.END)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.status_bar.config(text=f"Saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {str(e)}")
    
    def run_code(self):
        """Run the current code."""
        code = self.code_editor.get(1.0, tk.END)
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Running code...\n")
        
        # Run code in a separate thread
        threading.Thread(target=self.execute_code, args=(code,), daemon=True).start()
    
    def execute_code(self, code):
        """Execute code and capture output."""
        try:
            # Redirect stdout to capture output
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Execute the code
            exec(code)
            
            # Get the output
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            
            # Update UI in main thread
            self.root.after(0, self.update_output, output or "Code executed successfully (no output)")
            
        except Exception as e:
            self.root.after(0, self.update_output, f"Error: {str(e)}")
    
    def update_output(self, output):
        """Update the output display."""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, output)
    
    def analyze_code(self):
        """Manually analyze code."""
        code = self.code_editor.get(1.0, tk.END)
        if code.strip():
            self.analyze_code_background(code)
        else:
            messagebox.showinfo("Info", "No code to analyze")
    
    def show_code_generator(self):
        """Show code generation dialog."""
        dialog = CodeGeneratorDialog(self.root, self.analyzer)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.code_editor.insert(tk.INSERT, dialog.result)
    
    def suggest_fixes(self):
        """Suggest fixes for current code."""
        code = self.code_editor.get(1.0, tk.END)
        
        # Get current analysis
        analysis = self.analyzer.analyze_syntax(code)
        
        if analysis["syntax_errors"]:
            error_msg = analysis["syntax_errors"][0]["message"]
            
            # Show loading message
            self.analysis_text.insert(tk.END, "\nGenerating fix suggestions...\n")
            
            # Generate fixes in background
            threading.Thread(
                target=self.generate_fixes_background, 
                args=(code, error_msg), 
                daemon=True
            ).start()
        else:
            messagebox.showinfo("Info", "No syntax errors found to fix")
    
    def generate_fixes_background(self, code, error_msg):
        """Generate fixes in background."""
        fixes = self.analyzer.suggest_fixes(code, error_msg)
        self.root.after(0, self.show_fixes, fixes)
    
    def show_fixes(self, fixes):
        """Show generated fixes."""
        self.analysis_text.insert(tk.END, f"\nSuggested fixes:\n{fixes}\n")

class CodeGeneratorDialog:
    def __init__(self, parent, analyzer):
        """Initialize code generator dialog."""
        self.analyzer = analyzer
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("AI Code Generator")
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_dialog()
    
    def setup_dialog(self):
        """Set up the dialog interface."""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Description input
        ttk.Label(main_frame, text="Describe what you want to code:").pack(anchor=tk.W)
        self.description_text = scrolledtext.ScrolledText(main_frame, height=5)
        self.description_text.pack(fill=tk.X, pady=(5, 10))
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="python")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, 
                                 values=["python", "javascript", "java", "cpp"])
        lang_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Generated code display
        ttk.Label(main_frame, text="Generated Code:").pack(anchor=tk.W)
        self.generated_text = scrolledtext.ScrolledText(main_frame, height=15)
        self.generated_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Generate", command=self.generate_code).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Insert", command=self.insert_code).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT)
    
    def generate_code(self):
        """Generate code based on description."""
        description = self.description_text.get(1.0, tk.END).strip()
        language = self.language_var.get()
        
        if not description:
            messagebox.showwarning("Warning", "Please enter a description")
            return
        
        self.generated_text.delete(1.0, tk.END)
        self.generated_text.insert(tk.END, "Generating code...")
        
        # Generate in background
        threading.Thread(
            target=self.generate_background, 
            args=(description, language), 
            daemon=True
        ).start()
    
    def generate_background(self, description, language):
        """Generate code in background."""
        code = self.analyzer.generate_code(description, language)
        self.dialog.after(0, self.update_generated_code, code)
    
    def update_generated_code(self, code):
        """Update generated code display."""
        self.generated_text.delete(1.0, tk.END)
        self.generated_text.insert(tk.END, code)
    
    def insert_code(self):
        """Insert generated code."""
        code = self.generated_text.get(1.0, tk.END).strip()
        if code:
            self.result = code
            self.dialog.destroy()
        else:
            messagebox.showwarning("Warning", "No code to insert")
    
    def cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()

def main():
    """Main function to run the AI-assisted coding environment."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("AI features will not work without a valid API key.")
    
    root = tk.Tk()
    app = AICodeEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()

