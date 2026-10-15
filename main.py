"""Single entry point:  python main.py <command>"""
import argparse


def main():
    p = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis")
    p.add_argument("command", choices=["explore"])
    args = p.parse_args()

    if args.command == "explore":
        from src.visualize import show_sample_images, show_preprocessing
        show_sample_images()
        show_preprocessing()
        print("Saved sample_images.png and preprocessing.png in outputs/")


if __name__ == "__main__":
    main()