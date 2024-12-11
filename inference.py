from roboflow import Roboflow

def get_model(model_id):
    # Initialize the Roboflow API with your API key
    rf = Roboflow(api_key="D19FVfXVsgB2ZEVrOP6F")  # Replace with your actual API key

    # Use the correct workspace name and model ID
    workspace_name = "team3-ohd3p"
    model = rf.workspace(workspace_name).project(model_id).version(4).model
    return model
