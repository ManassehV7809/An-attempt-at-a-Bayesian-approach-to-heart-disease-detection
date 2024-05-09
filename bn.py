import sys
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import networkx as nx
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from pgmpy.estimators import BayesianEstimator
from sklearn.preprocessing import KBinsDiscretizer

# Fetching dataset
print("Fetching dataset.")
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
data = pd.concat([X, y], axis=1)

#Droping missing values
data.dropna(inplace=True)

# Continuous features to discretize
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Using KBinsDiscretizer to discretize variables
discretizer = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
data_transformed = discretizer.fit_transform(data[continuous_features])
data_binned = pd.DataFrame(data_transformed, columns=[f"{col}_bin" for col in continuous_features], index=data.index)
data = pd.concat([data.drop(columns=continuous_features), data_binned], axis=1)


#Getting the name of the target for later use
target_variable = data.columns[-1]

# Building model with K-fold cross-validation
n_splits = 5  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
target_variable = data.columns[-1]

# Initialize lists to store metrics for each fold
fold_metrics = []

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f'\nFold: {fold+1}')

    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

    # Bayes model building  
    bic_score = BicScore(data=train_data)
    hc_search = HillClimbSearch(data=train_data)
    estimated_model = hc_search.estimate(scoring_method=bic_score, max_indegree=5, tabu_length=7)
    model = BayesianModel(estimated_model.edges())
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=1)

    #Saving the learnt BN as a pdf for visualization
    print("Visualizing Bayesian Network.")
    nx_graph = nx.DiGraph(model.edges())
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(12, 12))
    nx.draw(nx_graph, pos, with_labels=True, node_size=5000, node_color='skyblue', edge_color='#424242', font_weight='bold', font_size=10)
    plt.title('Learned Bayesian Network')
    plt.savefig('learned_bn_structure.pdf', format='pdf', dpi=300)
    print("Visualization saved as 'learned_bn_structure.pdf'.")

    # Evaluate model on test data
    inference = VariableElimination(model)
    predictions = []
    for _, instance in test_data.iterrows():
        evidence_dict = {var: instance[var] for var in model.nodes() if var != target_variable}
        pred = inference.map_query(variables=[target_variable], evidence=evidence_dict)[target_variable]
        predictions.append(pred)
    #modifying labels to allow for binary classification
    true_labels_binary = (test_data[target_variable] != 0).astype(int)
    predictions_binary = [1 if pred != 0 else 0 for pred in predictions]

    # Calculate evaluation metrics for the fold
    accuracy = accuracy_score(true_labels_binary, predictions_binary)
    precision = precision_score(true_labels_binary, predictions_binary)
    recall = recall_score(true_labels_binary, predictions_binary)
    f1 = f1_score(true_labels_binary, predictions_binary)
    conf_matrix = confusion_matrix(true_labels_binary, predictions_binary)

    # Store fold results
    fold_metrics.append({
        'Fold': fold+1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    })

    # Print metrics for this fold
    print(f"Accuracy (Fold {fold+1}): {accuracy:.4f}")
    print(f"Precision (Fold {fold+1}): {precision:.4f}")
    print(f"Recall (Fold {fold+1}): {recall:.4f}")
    print(f"F1 Score (Fold {fold+1}): {f1:.4f}")
    print(f"Confusion Matrix (Fold {fold+1}):\n{conf_matrix}")

# Calculate and print average metrics across all folds
avg_accuracy = sum(metric['Accuracy'] for metric in fold_metrics) / n_splits
avg_precision = sum(metric['Precision'] for metric in fold_metrics) / n_splits
avg_recall = sum(metric['Recall'] for metric in fold_metrics) / n_splits
avg_f1 = sum(metric['F1 Score'] for metric in fold_metrics) / n_splits

# Save outputs to a text file
with open('output_metrics.txt', 'w') as f:
    sys.stdout = f  # Redirect stdout to the file
    print("Metrics for Each Fold:")
    for metric in fold_metrics:
        print(f"\nFold: {metric['Fold']}")
        print(f"Accuracy: {metric['Accuracy']:.4f}")
        print(f"Precision: {metric['Precision']:.4f}")
        print(f"Recall: {metric['Recall']:.4f}")
        print(f"F1 Score: {metric['F1 Score']:.4f}")
        print(f"Confusion Matrix:\n{metric['Confusion Matrix']}")

    print("\nAverage Metrics Across all Folds:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

# Reset stdout
sys.stdout = sys.__stdout__

print("Output metrics saved in 'output_metrics.txt'")
