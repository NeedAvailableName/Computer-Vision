
# Computer Vision project - Vietnamese Money Recognition

This final project's aim is to correctly classify Vietnamese currency based on VGG16 and Transfer Learning.



## Contributing

- Data collection and preprocessing: All members
- Feature extraction and model design: Biên & Đăng
- Model evaluation and report: Lâm & Mạnh


## Installation

1. First, you need to set up Python environment in your computer.

- Download Python and related tools here https://www.python.org/downloads/
2. Clone this repo
```bash
git clone https://github.com/NeedAvailableName/Computer-Vision.git
cd vietnamese_currency_recognition
```
    
## Set up Dataset and Weight

To run this project, you will need to download and unzip the dataset and the weight of model to above directory.

Link to dataset: 
`API_KEY`

Link to weight:
`ANOTHER_API_KEY`

After extracting to folder, we should have a folder like this
![GitHub Logo](https://github.com/NeedAvailableName/Computer-Vision/blob/master/Screenshot%202024-02-02%20115545.png?raw=true)



## Deployment
- Install necessary package
```bash
pip install requirements.txt
```
- To train the model run
```bash
python train.py
```
- To test this project run
```bash
python money_recognition.py
```
- To evaluate the model run
```bash
python evaluate_model.py
```
