# print(">>> RUN.PY IS RUNNING <<<")


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    print("🚀 Starting full ML pipeline...")

    # 1️⃣ Data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print("✅ Data ingestion completed!")

    # 2️⃣ Data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    print(f"✅ Data transformation completed! Preprocessor saved at: {preprocessor_path}")

    # 3️⃣ Model training
    model_trainer = ModelTrainer()
    r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("\n================ Training Summary ================")
    print("✅ Training complete!")
    print(f"📈 Final R² Score: {r2:.4f}")
    print("💾 Model saved at: artifacts/model.pkl")
    print("=================================================\n")

if __name__ == "__main__":
    main()
