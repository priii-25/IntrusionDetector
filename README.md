# Network Intrusion Detection System

This is a Streamlit-based web application designed to detect network intrusions using machine learning. It leverages the Random Forest Classifier to analyze network traffic data, based on the KDD Cup 1999 dataset structure, to classify connections as normal or malicious (attacks). The application supports data exploration, model training, real-time prediction, and batch prediction, making it a comprehensive tool for network security analysis.

---

## Features

- **Data Exploration**: Visualize distributions, correlations, and feature characteristics of network traffic data.
- **Model Training**: Train a Random Forest classifier for either binary (Normal vs. Attack) or multiclass (specific attack types) classification. Evaluate model performance with metrics, confusion matrices, and feature importance.
- **Real-time Prediction**: Analyze individual network connections by inputting feature values to classify them as normal or attack.
- **Batch Prediction**: Process multiple connections from a file for bulk analysis and export results.
- **Help & Documentation**: Detailed guidance on data format, feature descriptions, and FAQs.

---

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.8+**
- Required Python packages (listed in `requirements.txt`):
  ```
  streamlit
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  joblib
  ```

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/network-intrusion-detection.git
   cd network-intrusion-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit server, and you can access the application in your web browser at `http://localhost:8501`.

---

## Usage

1. **Load Data**:
   - In the sidebar, upload a CSV or TXT file in the KDD'99 format (41 features + 1 attack label, no header) or use the built-in sample data.
   - Click **Load Data** to preprocess and load the dataset.

2. **Data Exploration**:
   - Navigate to the **Data Exploration** page to view dataset statistics, attack distributions, feature correlations, and individual feature distributions.

3. **Model Training**:
   - Go to the **Model Training** page.
   - Select classification type (Binary or Multiclass), test set size, number of trees, and features.
   - Click **Train Model** to train the Random Forest classifier.
   - Review performance metrics and download the trained model and scaler.

4. **Real-time Prediction**:
   - On the **Real-time Prediction** page, input values for the features used during training.
   - Submit to classify the connection as normal or attack, with confidence scores and probability visualizations.

5. **Batch Prediction**:
   - On the **Batch Prediction** page, upload a file with multiple connections (same format as training data).
   - Run predictions and view summary statistics, visualizations, and detailed results. Download filtered results as CSV.

6. **Help**:
   - Refer to the **Help** page for data format requirements, feature descriptions, and FAQs.

---

## Data Format

The application expects data in the **KDD Cup 1999** format:
- **File Type**: CSV or TXT
- **Header**: No header row
- **Columns**: 41 features + 1 attack label (e.g., `normal`, `neptune`, `smurf`)
- **Categorical Features**: `protocol_type`, `service`, `flag` should be text-based (e.g., `tcp`, `http`, `SF`)
- **Note**: Ensure attack labels do not have trailing dots (e.g., `normal.` should be `normal`).

For reference, see the [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

---

## Project Structure

- `app.py`: Main application script containing the Streamlit app logic.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file, providing an overview and instructions.

---

## Notes

- **Performance**: The application is optimized for the KDD'99 dataset. Performance on real-world network traffic may vary and requires validation.
- **Scalability**: For large datasets, training and prediction may take time. Adjust the number of trees and test set size accordingly.
- **Error Handling**: The app includes robust error handling for data loading, preprocessing, and prediction. Check the error messages for guidance.
- **Extensibility**: You can extend the app by adding new models, features, or visualizations by modifying `app.py`.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows the existing style and includes appropriate documentation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Disclaimer

This application is for **educational and illustrative purposes only**. It is not a replacement for professional network security solutions. Always validate and tune models for real-world applications.

---

## Contact

For questions or feedback, please open an issue on the GitHub repository or contact the maintainer at [your-email@example.com].

Happy analyzing! 🔒
