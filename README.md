# 🛒 Online Shoppers Purchase Prediction
### Marketing Optimization with a Full Machine Learning Lifecycle

## 📌 Project Overview
This project analyzes and models online user behavior to predict purchasing intention and support the optimization of digital marketing campaigns.

Using the Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository), the project:
- Reproduces the methodology of the original academic paper
- Extends it with modern machine learning engineering best practices
- Delivers an end-to-end ML system, from data analysis to API deployment
The final output is a production-ready ML pipeline exposed via a FastAPI service.

## 🎯 Objectives
- Understand customer online behavior through EDA and statistical analysis
- Identify key behavioral factors influencing purchase decisions
- Build predictive classification models for purchase intention
- Reproduce baseline results from the original publication
- Design a robust, reusable, and deployable ML pipeline
- Serve predictions through a clean, validated REST API

## 🔁 Machine Learning Lifecycle (Checklist)
This project follows a complete, real-world ML workflow:
✅ Business understanding & data exploration
✅ Data validation & preprocessing
✅ Feature engineering & MRMR selection
✅ Model training & evaluation
✅ Reproducible ML pipelines
✅ Model packaging & versioning
✅ FastAPI deployment
✅ Input validation, logging & error handling
✅ Automated testing (data, pipeline, API)


## 🔜 Roadmap / Future Improvements
- Monitoring post-deployment (data drift, performance)
- Experiment tracking & hyperparameter logging (MLflow, W&B)
- CI/CD pipeline integration
- Load testing & performance benchmarking


## Run the ML Project?

1. Install dependencies
```bash
   pip install -r requirements.txt
```

2. Run automated tests
```bash
pytest -v
```

3. Run the training script
```bash
python train.py
```

4. Evaluate the model
```bash
python evaluate.py
```

5. Run the FastAPI server
```bash
uvicorn app.main:app --reload
```
Open your browser at: http://127.0.0.1:8000/docs

This will start the FastAPI development server with auto-reload enabled.


## Project Structure

```bash
.
├─ app                                       # FastAPI / serving
│  ├─ main.py
│  ├─ model_loader.py
│  └─ schemas.py 
├─ data                                      # raw data & references
│  ├─ raw
│     └─ online_shoppers_intention.csv
│  └─ sakar2018.pdf             
├─ images                                    # visual assets 
├─ logs                                      # runtime logs   
├─ models                                    # trained models    
├─ notebook                                  # experimentation
│  └─ notebook_ospi-moai.ipynb    
├─ reports                                   # outputs, metrics, summaries
├─ src                                       # ML & core logic
│  ├─ __init__.py
│  ├─ data_loader.py
│  ├─ evaluation.py
│  ├─ logger.py
│  ├─ model_registry.py
│  ├─ models.py
│  ├─ pipeline.py
│  ├─ preprocessing.py
│  ├─ reporting.py
│  └─ validation.py  
├─ tests
│  ├─ test_api.py
│  ├─ test_data.py
│  └─ test_pipeline.py       
├─ config.py          
├─ conftest.py           
├─ train.py
├─ evaluate.py
├─ requirements.txt
└─ README.md
```

