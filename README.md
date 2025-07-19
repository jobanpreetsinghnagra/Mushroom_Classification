# 🍄 Mushroom Classification Web App

A **Streamlit** application that predicts whether a mushroom is **edible** or **poisonous** using **Logistic Regression**, trained on the UCI Mushroom Dataset.

---

## ⚙️ Technologies Used

- **Streamlit** – for building an interactive frontend UI  
- **scikit-learn** – for encoding, preprocessing, and logistic regression modeling  
- **joblib** – for saving and loading the model and encoders  
- **pandas**, **numpy** – for data manipulation and numerical operations  

---

## 📁 Project Structure

```text
MUSHROOM_CLASSIFICATION/
├── data/
│   ├── labels.csv                 # Encoded Y values (target labels)
│   ├── mushrooms.csv              # Raw mushroom dataset
│   └── parameters.csv             # Processed X values (features)
├── models/
│   ├── feature_names.pkl          # Saved feature names for display/input
│   ├── mushroom_model.pkl         # Trained Logistic Regression model
│   ├── ordinal_encoder.pkl        # Encoder for input features
│   └── target_encoder.pkl         # Encoder for output labels
├── pages/
│   ├── Classifier.py              # Main mushroom classifier UI
│   ├── guide.py                   # User guide/help section
│   └── Welcome.py                 # Welcome/Landing page
├── tests/
│   ├── edible.txt                 # Sample edible output
│   └── poisonous.txt              # Sample poisonous output
├── data.ipynb                     # Notebook for exploring dataset
├── model.ipynb                    # Jupyter notebook for data prep & model training
├── streamlit_app.py               # Entry point for the Streamlit app
├── requirements.txt               # List of dependencies
├── README.md                      # Project documentation
├── LICENSE                        # Project license
└── todo.txt                       # Development tasks and notes
```

---

## 🧠 How the Model Works

1. **Dataset**: UCI Mushroom Dataset
2. **Preprocessing**:

   * Handle missing values (e.g., '?')
   * Encode categorical features using `OrdinalEncoder`
3. **Train/Test Split**: 80% for training, 20% for testing
4. **Model**: `LogisticRegression()` from scikit-learn
5. **Saving Artifacts**:

   * `mushroom_model.pkl`: trained model
   * `ordinal_encoder.pkl`: feature encoder
   * `target_encoder.pkl`: label encoder
   * `feature_names.pkl`: used in the Streamlit UI

---

## 🚀 Running the Project Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/MUSHROOM_CLASSIFICATION.git
   cd MUSHROOM_CLASSIFICATION
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model training notebook**
   Open and execute all cells in `data.ipynb`  `model.ipynb` to:

   * Prepare the dataset
   * Train the model
   * Generate and save encoders and model artifacts

4. **Launch the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   The web app will open in your default browser, typically at `http://localhost:8501`.

---

## 🧭 App Pages (via Streamlit Multipage)

* **Welcome** – Introduction and instructions
* **Guide** – How to use the classifier
* **Classifier** – Interactive prediction interface

---

## 💡 Possible Enhancements

* Add model evaluation metrics in the app (accuracy, precision, confusion matrix)
* Allow users to upload CSV files for batch classification
* Add support for other ML models (e.g., Random Forest, SVM)
* Deploy the app using Streamlit Cloud or Docker

---

## 🛠️ Contribution

Found a bug or want to improve something? Feel free to:

* **Open an Issue**
* **Submit a Pull Request**

You can also reach out via email for direct support or suggestions.

---

## 📜 License

This project is licensed under the terms of the **MIT License**. See the `LICENSE` file for more details.

---

Let me know if you’d like help generating a badge section (for GitHub Actions, license, etc.) or setting up deployment instructions.
