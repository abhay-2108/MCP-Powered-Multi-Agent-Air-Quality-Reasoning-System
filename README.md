#  MCP-Powered Multi-Agent Air Quality Reasoning System

A deep learning–based air quality analysis system exposed through the **Model Context Protocol (MCP)**. Four specialized neural network models work together as MCP tools, enabling LLM agents to reason about air quality from multiple perspectives — pollutant data, satellite/street imagery, vehicle emissions, and public health risk.

---

##  Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    LLM Agent (Client)                    │
│         Receives user query → selects tools →            │
│         chains results → delivers final answer           │
└────────────────────────┬─────────────────────────────────┘
                         │ MCP Protocol (stdio)
┌────────────────────────▼─────────────────────────────────┐
│                   MCP Server (server.py)                 │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  AQI LSTM    │  │  CNN Image   │  │  Emission    │    │
│  │  Forecaster  │  │  Classifier  │  │  Predictor   │    │
│  │  (Keras)     │  │  (ONNX)      │  │  (ONNX)      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                    ┌──────────────┐                      │
│                    │  Health Risk │                      │
│                    │  Estimator   │                      │
│                    │  (ONNX)      │                      │
│                    └──────────────┘                      │
└──────────────────────────────────────────────────────────┘
```

---

##  MCP Tools

### 1. `predict_aqi_forecast`
> Predicts AQI for the next **4 hours** using an LSTM model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pm25`, `pm10` | float | Particulate matter concentrations |
| `no`, `no2`, `nox` | float | Nitrogen oxide levels |
| `nh3`, `co`, `so2`, `o3` | float | Other gas concentrations |
| `benzene`, `toluene`, `xylene` | float | Volatile organic compounds |

**Returns:** Current AQI estimate + 4-hour forecast.

---

### 2. `classify_pollution_image`
> Classifies air pollution severity from an urban photograph using a CNN (ResNet-based).

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | string | Absolute path to an urban/outdoor image |

**Returns:** Pollution category — *Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Severe*.

---

### 3. `predict_vehicle_emission`
> Predicts vehicle emission level (Low / Medium / High) based on vehicle and driving characteristics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_size` | float | *required* | Engine displacement (L) |
| `mileage` | float | *required* | Total distance driven (km) |
| `speed` | float | *required* | Current speed (km/h) |
| `vehicle_type` | string | `"Car"` | Bus, Car, Motorcycle, Truck |
| `fuel_type` | string | `"Gasoline"` | Diesel, Electric, Gasoline, Hybrid |
| `road_type` | string | `"Highway"` | City, Highway, Rural |
| `traffic` | string | `"Moderate"` | Low, Moderate, Heavy |
| `age` | int | `5` | Vehicle age (years) |
| `acceleration` | float | `2.5` | Acceleration (m/s²) |
| `temperature` | float | `25` | Ambient temperature (°C) |
| `humidity` | float | `50` | Relative humidity (%) |
| `wind_speed` | float | `10` | Wind speed (km/h) |
| `air_pressure` | float | `1013` | Atmospheric pressure (hPa) |

**Returns:** Emission category and confidence probabilities.

---

### 4. `estimate_health_risk`
> Estimates public health impact using a multi-task model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `aqi` | float | Current Air Quality Index |
| `pm10`, `pm25` | float | Particulate matter levels |
| `no2`, `so2`, `o3` | float | Gas concentrations |
| `temperature` | float | Ambient temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `wind_speed` | float | Wind speed (km/h) |

**Returns:** Health impact score (0–100), risk category, and recommendation.

---



##  Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/abhay-2108/MCP-Powered-Multi-Agent-Air-Quality-Reasoning-System.git
cd MCP-Powered-Multi-Agent-Air-Quality-Reasoning-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install mcp[cli] tensorflow onnxruntime numpy Pillow
```

### 4. Run the MCP Server

```bash
python mcp_server/server.py
```

The server starts via **stdio** transport — it's designed to be launched by an MCP-compatible client (like Gemini CLI, Claude Desktop, or a custom agent).

### 5. Client Configuration

Add the following to your MCP client's config (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "air-quality": {
      "command": "./venv/Scripts/python.exe",
      "args": ["mcp_server/server.py"],
      "env": { "PYTHONPATH": "." }
    }
  }
}
```

---

##  Example Usage

See [`sample_input.txt`](sample_input.txt) for a full set of example prompts. Here's a quick example:

**Prompt:**
> Predict the air quality index for the next 4 hours given PM2.5 = 85, PM10 = 110, NO = 15, NO2 = 43, NOx = 55, NH3 = 8, CO = 1.2, SO2 = 10, O3 = 62, Benzene = 2.1, Toluene = 5.4, Xylene = 1.8

**Response:**
| Timeframe | AQI |
|-----------|-----|
| Current | 128.1 |
| +1 hour | 139.3 |
| +2 hours | 141.7 |
| +3 hours | 144.4 |
| +4 hours | 148.0 |

---

##  Models

| Model | Architecture | Format | Task |
|-------|-------------|--------|------|
| AQI Forecaster | LSTM (stacked) | Keras | 4-hour AQI time series prediction |
| Image Classifier | ResNet-50 (fine-tuned) | ONNX | 6-class air pollution severity from photos |
| Emission Predictor | Dense NN | ONNX | Vehicle emission level (Low/Medium/High) |
| Health Estimator | Multi-task Dense NN | ONNX | Health impact score + risk category |

All models were trained in the Jupyter notebooks under `Notebooks/`. Training plots and visualizations are saved in `Images/`.

---

##  Multi-Agent Reasoning

The system is designed for **chained reasoning** — an LLM agent can:

1. **See** → Classify pollution from a camera image
2. **Source** → Estimate vehicle emission contributions
3. **Forecast** → Predict AQI trends for the next 4 hours
4. **Assess** → Evaluate public health risk and generate advisories

The agent dynamically selects and sequences tools based on the user's query, combining results into a unified analysis.

---

