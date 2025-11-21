import nbformat
import pickle
from nbconvert import PythonExporter


def extract_model_from_notebook():
    try:
        # Read the notebook
        with open('../main_project.ipynb') as f:
            nb = nbformat.read(f, as_version=4)

        # Convert notebook to python
        python_exporter = PythonExporter()
        python_code, _ = python_exporter.from_notebook_node(nb)

        # Write the python code to a temporary file
        with open('temp_notebook.py', 'w') as f:
            f.write(python_code)

        # Create a new namespace to execute the code
        namespace = {}

        # Execute the notebook code
        exec(python_code, namespace)

        # Look for common model variable names
        model_names = ['model', 'clf', 'classifier', 'trained_model']
        found_model = None

        for name in model_names:
            if name in namespace:
                found_model = namespace[name]
                break

        if found_model is None:
            print("Could not find the model variable in the notebook.")
            print("Available variables:", list(namespace.keys()))
            return False

        # Save the model
        with open('model.pkl', 'wb') as f:
            pickle.dump(found_model, f)

        print("Model successfully saved as model.pkl")
        return True

    except Exception as e:
        print(f"Error extracting model: {str(e)}")
        return False


if __name__ == "__main__":
    print("Extracting model from notebook...")
    extract_model_from_notebook()
