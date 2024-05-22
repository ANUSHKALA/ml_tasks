from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import wandb
import optuna
from PIL import Image as im

wandb.init(
    project="mlops_task",
    config={
        "learning_rate": 0.04,
        "epochs": 50
    }
)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Pastel1")
    plt.title("Confusion matrix", size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45, size=10)
    plt.yticks(tick_marks, [str(i) for i in range(10)], size=10)
    plt.tight_layout()
    plt.ylabel("Actual label", size=15)
    plt.xlabel("Predicted label", size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(
                str(cm[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )
    wandb.log({"Confusion Matrix Plot": plt})
    plt.close()


def get_hyper_params_from_optuna(trial):
    penality = trial.suggest_categorical("penality", ["l1", "l2", "elasticnet", None])

    if penality == None:
        solver_choices = ["newton-cg", "lbfgs", "sag", "saga"]
    elif penality == "l1":
        solver = "liblinear"
    elif penality == "l2":
        solver_choices = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    elif penality == "elasticnet":
        solver = "saga"

    if not (penality == "l1" or penality == "elasticnet" ):
        if penality == None:
            solver = trial.suggest_categorical("solver_" + "None", solver_choices)
        else:
            solver = trial.suggest_categorical("solver_" + penality, solver_choices)

    C = trial.suggest_float("inverse_of_regularization_strength", 0.1, 1)

    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)

    if penality == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    else:
        l1_ratio = None
    return penality, solver, C, fit_intercept, intercept_scaling, l1_ratio

# def get_hyper_params_from_optuna(trial):
#     penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])

#     if penalty == "none":
#         solver = "sag"
#         C = trial.suggest_float("inverse_of_regularization_strength", 0.1, 1)
#         fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
#         intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)
#         l1_ratio = None
#     else:
#         if penalty == "elasticnet":
#             solver = "saga"
#         else:
#             solver_choices = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
#             solver = trial.suggest_categorical("solver_" + penalty, solver_choices)
#         C = trial.suggest_float("inverse_of_regularization_strength", 0.1, 1)
#         fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
#         intercept_scaling = trial.suggest_float("intercept_scaling", 0.1, 1.0)
#         if penalty == "elasticnet":
#             l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
#         else:
#             l1_ratio = None

#     return penalty, solver, C, fit_intercept, intercept_scaling, l1_ratio


def sanity_checks(digits):
    print(digits.data.shape)
    print(digits.target.shape)
    wandb.log({"Image Data Shape": digits.data.shape})
    wandb.log({"Label Data Shape": digits.target.shape})

    plt.figure(figsize=(20, 4))

    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
        plt.title(f"Training: {label}", fontsize=20)

    wandb.log({"Sanity checks plot": plt})
    plt.close()


def visualize_test(x_test, y_test, predictions):
    table = wandb.Table(columns=["image", "label", "prediction"])
    for image_array, label, prediction in zip(x_test, y_test, predictions):
        # Normalize image array to the range [0, 1]
        if image_array.max() > 1.0:
            image_array = image_array / 16.0  # Since digit images are in range [0, 16]

        # Scale the image array to the range [0, 255]
        image_array = (image_array * 255).astype(np.uint8)

        # Reshape the image array to (8, 8)
        image_array = np.reshape(image_array, (8, 8))

        image = im.fromarray(image_array, mode="L")
        table.add_data(wandb.Image(image), label, prediction)
    wandb.log({"Prediction Table": table})



def evaluate_model(logisticRegr, x_test, y_test, predictions):
    score = logisticRegr.score(x_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')  # Ensure precision is calculated correctly
    recall = recall_score(y_test, predictions, average='macro')

    wandb.log({
        "Mean Accuracy": score,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,  # Log precision as a number
        "Recall": recall
    })
    print("score: ",score, "balanced_acc: ", balanced_accuracy, "precision: ",precision,"recall: ",recall)

    return score, balanced_accuracy, precision, recall



def show_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    wandb.log({"Confusion Matrix": cm})
    plot_confusion_matrix(cm)


def objective(trial):
    digits = datasets.load_digits()

    sanity_checks(digits)

    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=0
    )

    (
        penalty,
        solver,
        C,
        fit_intercept,
        intercept_scaling,
        l1_ratio,
    ) = get_hyper_params_from_optuna(trial)

    logisticRegr = LogisticRegression(
        penalty=penalty,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        solver=solver,
        l1_ratio=l1_ratio,
        max_iter=1000
    )
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)

    visualize_test(x_test, y_test, predictions)

    score, balanced_accuracy, precision, recall = evaluate_model(
        logisticRegr, x_test, y_test, predictions
    )

    show_confusion_matrix(y_test, predictions)

    return balanced_accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    trial = study.best_trial

    print(f"Balanced Accuracy: {trial.value}")
    print(f"Best hyperparameters: {trial.params}")


main()
