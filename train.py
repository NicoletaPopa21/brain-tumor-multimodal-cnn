import tensorflow as tf

# Import custom modules from the 'src' folder
from src.data_loader import download_and_prepare_data
from src.architecture import build_fractional_multimodal_model

def main():
    print("=== Multimodal Brain Tumor Classification Pipeline ===")
    
    #Load and process the MRI and CT data
    train_mri, test_mri, train_ct, test_ct, train_y, test_y = download_and_prepare_data()
    
    #Build the Dual-Branch Multimodal architecture
    print("Building the Multimodal Fractional CNN...")
    model = build_fractional_multimodal_model(input_shape=(224, 224, 3))
    model.summary()
    
    #Train the model
    print("Starting training process...")
    history = model.fit(
        [train_mri, train_ct], train_y,
        validation_data=([test_mri, test_ct], test_y),
        epochs=10,
        batch_size=32
    )
    
    #Evaluate the model on the test set
    print("Evaluating the final model on validation data...")
    loss, accuracy = model.evaluate([test_mri, test_ct], test_y, verbose=0)
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}")
    
    #Save the trained model
    model.save("multimodal_fractional_model.keras")
    print("Model saved successfully as 'multimodal_fractional_model.keras'")

if __name__ == "__main__":
    main()
