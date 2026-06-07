from src.data_loader import load_data
from src.model import build_model
from src.evaluate import Evaluate, Predict
from src.preprocessing import preprocessing
from src.train import Optimize_model
from src.visualizations import Accuracy_loss_visualization, Accuracy_epochs_visualization
from src.utils import set_seed

def main():
    set_seed(3)
    df = load_data()
    X_train_std, X_test_std, y_train, y_test = preprocessing(df)
    print("Data loaded and preprocessed successfully.")
    print(f"Training data shape: {X_train_std.shape}, Training labels shape: {y_train.shape},Test data shape: {X_test_std.shape}, Test labels shape: {y_test.shape}")
    model = build_model()
    print("Model built successfully.")
    history = Optimize_model(model, X_train_std, y_train, epochs=10)
    print("Model optimized successfully.")
    evaluate_results = Evaluate(model, X_test_std, y_test)
    print(f"Evaluation results: {evaluate_results}")
    Accuracy_loss_visualization(history)
    Accuracy_epochs_visualization(history)
    
if __name__ == "__main__":    main()
  
    
    
    
