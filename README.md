
# GazeGrip

**GazeGrip** is a Python-based application designed to enable users to control mouse input using eye-tracking technology. Built with the Kivy framework, it offers a cross-platform graphical interface that translates gaze movements into cursor actions, enhancing accessibility and providing alternative interaction methods.

## Features

- **Eye-Tracking Integration**: Utilizes eye-tracking data to control mouse movements.
- **Kivy Interface**: Employs Kivy for a responsive and touch-friendly GUI.
- **Customizable Settings**: Allows users to adjust sensitivity and calibration parameters.
- **Cross-Platform Support**: Compatible with multiple operating systems via Kivy.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Y0J7/GazeGrip.git
   cd GazeGrip
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

## File Structure

- `main.py`: Main application script.
- `style.kv`: Kivy layout definitions.
- `requirements.txt`: List of Python dependencies.
- `buildozer.spec`: Configuration for building mobile applications.
- `assets/`: Directory containing image assets like `icon.png` and `presplash.png`.

## Usage

Upon launching the application:

1. Calibrate your eye-tracking device as per the manufacturer's instructions.
2. The application window will display the GUI.
3. Your gaze will control the mouse cursor within the application window.
4. Adjust settings as needed to fine-tune responsiveness.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
