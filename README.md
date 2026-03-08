# AI-Driven Lane Detection and Navigation Path Prediction

This project implements a real-time lane detection and navigation path prediction system using computer vision and deep learning techniques.

## Features

- Real-time lane detection
- Path prediction (Left / Straight / Right)
- Steering arrow visualization
- Robust preprocessing for low-light conditions
- Streamlit demo interface

## Technologies Used

- Python
- OpenCV
- YOLOv8
- NumPy
- Streamlit

## Project Architecture

Video Input → Preprocessing → Lane Detection → Lane Geometry Analysis → Path Prediction → Output Visualization

## Installation

```bash
pip install -r requirements.txt

Run the project
```bash
python src/lane_detection.py

Demo
Run the web interface:
```bash
streamlit run app/streamlit_app.py

Applications

Autonomous vehicles

ADAS systems

Smart transportation

Road safety monitoring

---
## Run the Project

Run main program:
```bash
python src/lane_detection.py

Press ESC to stop.
