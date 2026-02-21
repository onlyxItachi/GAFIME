import argparse
import sys
from gafime.tutorial import generate_tutorial

def main():
    parser = argparse.ArgumentParser(description="GAFIME CLI - GPU-Accelerated Feature Interaction Mining Engine")
    parser.add_argument("-i", "--init", action="store_true", help="Generate an interactive starter Jupyter notebook (gafime_tutorial.ipynb)")
    parser.add_argument("-o", "--output", type=str, default="gafime_tutorial.ipynb", help="Output path for the generated tutorial notebook")

    args = parser.parse_args()

    if args.init:
        generate_tutorial(output_path=args.output)
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
