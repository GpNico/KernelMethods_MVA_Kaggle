import os

def main():
    os.system("python run.py -c pipelines/spectrumKernelRidgeNormalized.yml -o final_evaluation.csv")

if __name__ == "__main__":
    main()