import os
import argparse

from similarity_analyzer import TrajectorySimilarityAnalyzer


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description='Run trajectory similarity analysis')
    parser.add_argument('--input', '-i', default='data/data.csv')
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--resample', '-r', type=int, default=300)
    args = parser.parse_args()

    analyzer = TrajectorySimilarityAnalyzer(input_path=args.input, output_dir=args.output, config={'resample_points': args.resample})
    result = analyzer.run()
    print('Done. Output:', result['output_dir'])


if __name__ == '__main__':
    main()
