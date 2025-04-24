import datetime

# Generate a standardized name for ML models
def generate_model_name(project_name, model_arch, target_class=None, transfer_learning=False, epoch=None, batch_size=None, custom_tag=None, class_weight=False, train_transfer_model=False):
    """
    Generate a standardized name for ML models.
    Parameters:
    - project_name (str): Name of the project
    - model_arch (str): Architecture name (CNN, ResNet, etc.)
    - target_class (str, optional): Target class for binary classification
    - transfer_learning (bool, optional): Whether transfer learning was used
    - epoch (int, optional): Number of epochs
    - batch_size (int, optional): Batch size used for training
    - custom_tag (str, optional): Additional custom tag
    Returns:
    - str: Standardized model name
    """
    print("\n--Generating model name--")

    components = [project_name, model_arch]

    if target_class:
        components.append(f"bin_{target_class}")
    else:
        components.append("multi")

    if transfer_learning:
        components.append("transfer")

    if epoch is not None:
        components.append(f"e{epoch}")

    if batch_size is not None:
        components.append(f"b{batch_size}")

    if custom_tag:
        components.append(custom_tag)

    if class_weight:
        components.append("class_weight")

    if train_transfer_model:
        components.append("fine_tuning")

    model_name = "_".join(components)

    print(f"Generated model name: {model_name}")

    return model_name

# Generate the full path for the model
def get_model_path(model_name, models_folder):
    """
    Generate the full path for a model.
    Parameters:
    - model_name (str): Name of the model
    - models_folder (str): Path to the models folder
    Returns:
    - Full path to the model file
    """
    return f"{models_folder}/{model_name}.keras"


def generate_captioning_model_name(
    project_name,
    is_gru,
    attention,
    semantic_loss,
    reduce_lr_on_plateau
):
    """
    Generate a standardized name for image captioning models.
    """
    components = [project_name, "caption"]

    if is_gru:
        components.append("gru")
    else:
        components.append("lstm")

    if attention and not is_gru:
        components.append("attn")
    else:
        components.append("basic")

    if semantic_loss:
        components.append("sem_loss")
    else:
        components.append("basic_loss")

    if reduce_lr_on_plateau:
        components.append("reduce_lr")

    model_name = "_".join(components)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"{model_name}_{timestamp}"

    print(f"Generated model name: {model_name}")

    return model_name