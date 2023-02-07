ECG Analyzer

This project is made to classify ECG data. This algorithm can recognize normal sinus rhythm, atrial fibrillation (AF), an alternative rhythm, or is too noisy to be classified. More information about the data can be found here https://physionet.org/content/challenge-2017/1.0.0/.


If you want to use this algorithm, you need to:
1. Download files ECG_classificator.py and model.pt
2. Install required libraries
3. Run ECG_classificator.py and go to the web page
4. Normalize your data and upload it to the site (you can find an example of how the data should look like in the validation folder, 'data.csv').
5. Click "Analyze"

If you want to test the algorithm, you can do 1, 2, 3 from the previous point and use the data in the validation folder ('data.csv' - ECG data, 'target.csv' - correct answers).

In the file 'analyze_data.ipynb' you can find my data mining and algorithm development