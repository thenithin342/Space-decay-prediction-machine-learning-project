"""
Execute Jupyter Notebooks Programmatically
"""

import os
import sys
import subprocess

def run_notebook(notebook_path):
    """
    Execute a Jupyter notebook
    """
    print(f"\n{'='*80}")
    print(f"Executing: {notebook_path}")
    print('='*80)
    
    try:
        # Try using papermill if available
        result = subprocess.run(
            ['python', '-m', 'papermill', notebook_path, notebook_path, '--kernel', 'python3'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {notebook_path} executed successfully!")
            return True
        else:
            print(f"Papermill not available, trying nbconvert...")
            
            # Try nbconvert
            result = subprocess.run(
                ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', 
                 '--execute', '--inplace', notebook_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ {notebook_path} executed successfully!")
                return True
            else:
                print(f"⚠️  Could not execute automatically. Error: {result.stderr}")
                print(f"\nPlease run manually using:")
                print(f"  jupyter notebook {notebook_path}")
                return False
                
    except Exception as e:
        print(f"⚠️  Error: {str(e)}")
        print(f"\nPlease run the notebook manually:")
        print(f"  jupyter notebook {notebook_path}")
        return False

def main():
    """
    Execute all notebooks in order
    """
    notebooks = [
        'notebook/1_EDA_Analysis.ipynb',
        'notebook/2_Model_Training.ipynb'
    ]
    
    print("="*80)
    print("EXECUTING DATA SCIENCE NOTEBOOKS")
    print("="*80)
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            run_notebook(notebook)
        else:
            print(f"⚠️  Notebook not found: {notebook}")
    
    print("\n" + "="*80)
    print("NOTEBOOK EXECUTION SUMMARY")
    print("="*80)
    print("\nIf automatic execution didn't work, please run manually:")
    print("\n1. Open Jupyter Notebook:")
    print("   jupyter notebook")
    print("\n2. Navigate to notebook/ folder")
    print("\n3. Run notebooks in order:")
    print("   - 1_EDA_Analysis.ipynb")
    print("   - 2_Model_Training.ipynb")
    print("\nOr use Visual Studio Code with Jupyter extension!")

if __name__ == "__main__":
    main()

