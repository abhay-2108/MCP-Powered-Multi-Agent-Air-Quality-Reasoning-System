# MCP Implementation & Technical Novelty

This document outlines how the **Model Context Protocol (MCP)** transforms a collection of deep learning models into a unified, reasoning-capable AI system.

## 1. How MCP is Used in the Project

The project leverages the **FastMCP SDK** to build a modular server ([mcp_server/server.py](file:///P:/College%20Projects/Deep%20Learning/MCP-Powered%20Multi-Agent%20Air%20Quality%20Reasoning%20System/mcp_server/server.py)) that acts as a bridge between high-level reasoning and low-level computation.

### The Sensor-to-Reasoning Bridge
- **Tool Exposure:** Each deep learning model (LSTM, CNN, DNN) is wrapped in an `@mcp.tool()` decorator. This exposes them as capabilities that an LLM can discover and invoke dynamically.
- **Data Normalization:** The server handles all heavy lifting—image resizing, ImageNet normalization, tabular scaling, and feature engineering—within the tool logic. This keeps the LLM's workspace clean and focused on high-level decisions.
- **Heterogeneous Model Support:** MCP allows the system to run **Keras** (.keras) and **ONNX** (.onnx) models side-by-side seamlessly. The orchestrator doesn't need to know the underlying framework; it only interacts with the standardized MCP interface.

---

## 2. Why It Stands Out

Typical air quality systems provide a single metric (like AQI). This system stands out by providing **Contextual Intelligence**:

- **Multi-Modal Perception:** By combining **Computer Vision** (CNN) with **Time-Series Forecasting** (LSTM), the system can verify sensor data against visual evidence. If an image shows a thick haze but sensors report "Good" air, the system can flag a potential sensor malfunction.
- **Orchestrated Chaining:** The [orchestrator/agent.py](file:///P:/College%20Projects/Deep%20Learning/MCP-Powered%20Multi-Agent%20Air%20Quality%20Reasoning%20System/orchestrator/agent.py) demonstrates how tools are not just called in isolation but are **chained**. 
    1. *Vision Agent* detects haze.
    2. *Emission Agent* identifies the likely source (traffic).
    3. *Forecast Agent* predicts the duration.
    4. *Health Agent* assesses the risk to neighbors.

---

## 3. The Technical Novelty

The novelty of this research lies in moving from **"Single-Model Prediction"** to **"Multi-Agent Collaborative Reasoning."**

### i. Decoupled Intelligence Architecture
In traditional AI apps, the prediction logic is hardcoded. Here, the **Model Context Protocol** decouples the "thinking" (LLM) from the "expertise" (ONNX/Keras models).
- **Novelty:** You can swap a model for a better version (e.g., upgrade the CNN to a Vision Transformer) without changing a single line of the reasoning orchestrator.

### ii. Federated Model Execution
Different models have different requirements (GPU for CNN, CPU for tabular). By using MCP, these models can technically live on different servers while presenting a single, unified toolset to the agent.
- **Novelty:** This creates a **Federated Air Quality Intelligence** network where specialized models act as independent experts.

### iii. "Ground-Truth" Cross-Verification
The integration of visual air quality assessment (CNN) alongside chemical sensor data (LSTM) addresses a major gap in modern environmental monitoring: **verification.**
- **Novelty:** The system uses visual signatures as a secondary verification layer, adding a layer of reliability that sensor-only systems lack.

### iv. LLM-in-the-Loop Orchestration
Unlike static dashboards, this system uses an Agent to translate technical outputs (e.g., "Health Score: 0.82") into human-centric narrative advice. 
- **Novelty:** It transforms raw data into **Actionable Intelligence**, explaining *why* the air is bad and *what* the user should specifically do about it.
