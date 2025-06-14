from ultralytics import YOLO
import torch

def main():
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a pretrained YOLOv8 model (you can choose yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
    model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the nano model, good for quick experiments
    
    # Train the model
    results = model.train(
        data=r"C:\projectfolders\ParkAI\unified_dataset\data.yaml",  # Path to your dataset YAML file
        epochs=50,          # Number of epochs to train
        imgsz=640,          # Image size (YOLOv8 default is 640)
        batch=16,           # Batch size (adjust based on your GPU memory)
        device=device,      # Use GPU if available, else CPU
        seed=42,            # Random seed for reproducibility
        workers=4           # Number of dataloader workers (adjust if you face multiprocessing issues)
    )
    
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()
