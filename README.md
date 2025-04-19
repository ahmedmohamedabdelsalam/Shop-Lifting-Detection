# Shop Lifting Detection

This repository contains a complete pipeline for processing and analyzing surveillance video data to detect potential shoplifting behavior using computer vision and deep learning techniques. It includes video preprocessing, frame visualization, statistical analysis, and organized result saving.

## Features

- Loads and processes video data using OpenCV
- Preprocesses frames with resizing and normalization
- Analyzes frame statistics including brightness and motion
- Visualizes sampled frames with metadata
- Saves analysis results to JSON, CSV, and TXT formats
- Uses PyTorch and Torchvision for deep learning support
- Clean logging with Colorama and tqdm for progress bars

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ahmedmohamedabdelsalam/Shop-Lifting-Detection.git
cd Shop-Lifting-Detection
pip install -r requirements.txt
```

## Requirements

The main dependencies include:

- opencv-python>=4.5.0
- numpy>=1.19.0
- torch>=1.9.0
- torchvision>=0.10.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- tqdm>=4.65.0
- colorama>=0.4.6

Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Shop-Lifting-Detection/
│
├── data/                      # (Optional) Place to store input videos
├── results_*/                 # Generated result folders with timestamps
├── video_processor.py         # Main video processing class
├── main.py                    # Entry point for processing the dataset
├── requirements.txt           # Required packages
└── README.md                  # Project documentation
```

## Usage

Make sure your dataset is structured like this:

```
Shop DataSet/
├── shop lifters/
│   ├── video1.mp4
│   └── ...
├── non shop lifters/
│   ├── video2.mp4
│   └── ...
```

Run the main script:

```bash
python main.py
```

This will analyze all videos and save results and visualizations inside a timestamped `results_` folder.

## Output

The script will generate:

- PNG visualizations of sampled frames
- JSON and CSV files with frame statistics
- A summary report in text format

## License

This project is licensed under the MIT License.

---

Developed by [Ahmed Mohamed Abdelsalam](https://github.com/ahmedmohamedabdelsalam)
