
🌞 Solar Panel Dust Analysis & Automated Cleaning System

An AI-driven, IoT-enabled solar panel maintenance system that detects dust accumulation, automates water-efficient cleaning, and improves solar energy output. Designed for smart cities, municipal solar plants, and large-scale renewable energy deployments, this project addresses real-world efficiency loss using computer vision, embedded systems, and intelligent automation.

🚀 Problem Statement

Dust accumulation reduces solar panel efficiency by 20–30%, leading to significant power loss, increased maintenance costs, and water-intensive manual cleaning. Existing robotic cleaners are expensive and difficult to scale. A smart, automated, and cost-effective solution is required.

🎯 Objective

To develop an automated solar panel dust detection and cleaning system that:

Restores efficiency only when required

Reduces water consumption

Minimizes manual intervention

Scales for rooftop and large solar installations

💡 Solution Overview

The system uses light sensors and camera-based AI analysis to detect dust levels on solar panels. When dust exceeds a predefined threshold, an ESP32-controlled cleaning mechanism activates a mist spray and wiper to safely clean the panel. All results are visualized on a real-time dashboard.

🧠 Explainable AI (Grad-CAM + SHAP)

Every prediction is accompanied by an explanation, so operators can verify that the model is looking at the panel surface and not at background artefacts:

• Grad-CAM localization heatmaps — generated from the last convolutional block (`block5_conv3`) of the VGG16 encoder. For the deployed linear SVM the gradient of the decision score w.r.t. the pooled features is computed in closed form (`coef_ / scale`); RBF kernels are handled via the analytic Jacobian of the RBF decision function. The heatmap overlay shows exactly which surface regions drove the dust decision.
• SHAP feature attributions — offline Shapley values over the 512-dimensional GAP vector that feeds the SVM head, identifying the top contributing texture / colour channels per class.
• Confidence-gated review — samples classified below an 0.85 confidence threshold are flagged `requires_review: true` for human inspection and returned to the curation pool.

Endpoints

• `POST /analyze` — original dashboard endpoint. Returns `dustiness` percentage + confidence.
• `POST /explain` — new XAI endpoint. Returns probabilities, predicted class, confidence, `requires_review` flag, spatial-concentration of the localization mass and a base64 Grad-CAM overlay PNG.

CLI

    python explanations.py --image <panel_image.jpg>          # prints audit + saves Grad-CAM overlay

Training from scratch

    pip install -r requirements.txt
    # dataset layout:
    #   dataset/train/clean/*.jpg  dataset/train/dirty/*.jpg
    #   dataset/test/clean/*.jpg   dataset/test/dirty/*.jpg
    python train_solar_dust.py --data ./dataset --train-head

The training script extracts VGG16 GAP features, standardizes them, runs an early-pruned search over the SVM hyperparameters, saves the artifacts under `Models/` (`svm_classifier.pkl`, `scaler.pkl`, `class_names.json`, `pipeline_meta.json`) and generates all paper figures (`fig5_metrics.png`, `fig6_confusion_matrix.png`, `fig3_roc_auc.png`, `fig7_confidence.png`, `fig8_loss_accuracy.png`).

Public datasets to get started

• Kaggle – Solar Photovoltaics Panel for Dust Detection: https://www.kaggle.com/datasets/safwanshamsir99/solar-photovoltaics-panell-for-dust-dectection
• Kaggle – Solar Panel Images Clean & Faulty: https://www.kaggle.com/datasets/pythonafroz/solar-panel-images
• Kaggle – Solar Panel Dust Detection: https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection

⚙️ Key Features

• AI-based dust detection using VGG16 + SVM
• Sensor-based dust estimation using BH1750
• ESP32-controlled automated cleaning workflow
• Mist spray + mechanical wiper for water-efficient cleaning
• Real-time dashboard with analytics
• Environmental and financial impact estimation
• IoT-ready and cloud-scalable architecture
• Fallback mode if ML models are unavailable
• Explainable predictions (Grad-CAM heatmaps + SHAP attributions + review flags)

🧠 Technology Stack

Backend: Python, Flask, TensorFlow, scikit-learn, Joblib
Frontend: HTML5, CSS3, JavaScript, Tailwind CSS
AI/ML: VGG16 (feature extraction), SVM (classification), SHAP + Grad-CAM (explainable AI)
Hardware: ESP32, BH1750, Camera Module, Pump, Motor Driver, Wiper

🏗️ Working Principle

Sensors and camera detect dust accumulation

ESP32 evaluates dust threshold (>40%)

AI model predicts dust severity

Cleaning system activates mist spray + wiper

Dashboard updates efficiency and analytics

📊 Prototype Results

• ~25% improvement in solar panel efficiency
• ~40% reduction in water usage
• Fully automated operation
• Tested on a 60-cell solar panel

💰 Cost & Feasibility

Total prototype cost: ₹66,460
Commercial robotic cleaners cost ₹1–2 lakh per unit
Low maintenance, high scalability, and suitable for government deployment

🌱 Sustainability Impact

• Water conservation
• Reduced labor dependency
• Increased renewable energy output
• Supports India’s Smart City and Clean Energy missions

🏙️ Applications

Municipal solar plants, rooftop solar systems, government buildings, industrial solar farms, smart city infrastructure, and large-scale renewable installations.

🔮 Future Scope

AI-based predictive cleaning, cloud analytics, mobile monitoring app, real IoT sensor integration, multi-site solar management, automated alerts, and smart scheduling.

📁 Project Structure

Advanced solar panel project with Flask backend, ML models, modular frontend, IoT-ready static assets, and scalable architecture.

🏷️ Domain Classification

Domain Range: 29–99
Category: Renewable Energy | IoT | Mechatronics | Automation
Patent Range: Automated Solar Panel Cleaning Systems

🏆 Why This Project Stands Out

✔ Real-world problem solving
✔ AI + IoT + hardware integration
✔ Scalable and cost-effective design
✔ Government and industry relevance
✔ Strong sustainability focus

This project demonstrates end-to-end system design, combining AI, embedded systems, and renewable energy engineering into a deployable real-world solution.
