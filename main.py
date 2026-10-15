"""Single entry point:  python main.py <command>"""
import argparse


def main():
    p = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis")
    p.add_argument("command",
                   choices=["explore", "train", "evaluate", "predict", "grid"])
    p.add_argument("--image", help="path to X-ray image (for 'predict')")
    args = p.parse_args()

    if args.command == "explore":
        from src.visualize import show_sample_images, show_preprocessing
        show_sample_images()
        show_preprocessing()
        print("Saved sample_images.png and preprocessing.png in outputs/")
    elif args.command == "train":
        from src.train import train
        train()
    elif args.command == "evaluate":
        from src.evaluate import evaluate
        evaluate()
    elif args.command == "predict":
        from src.predict import predict_image
        label, conf, _ = predict_image(args.image)
        print(f"Prediction: {label}  (confidence {conf:.1%})")
        print("Reminder: educational demo, not a medical diagnosis.")
    elif args.command == "grid":
        from src.visualize import prediction_grid
        prediction_grid()


if __name__ == "__main__":
    main()