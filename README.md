# Multi-Agent Risk Intelligence Engine

---

# 🛡️ Aegis Risk Mesh: Multi-Agent Predictive Intelligence

**Aegis Risk Mesh** is a distributed, agentic ecosystem designed to perform autonomous risk assessment and predictive modeling. Unlike linear RAG pipelines, Aegis utilizes a **Stateful Agentic Mesh** powered by **LangGraph** and **Google ADK** to orchestrate specialized AI agents that negotiate outcomes through an **A2A (Agent-to-Agent)** protocol.

The system is optimized for **local-first inference** on the Mac M4 (via Ollama) and utilizes the **Model Context Protocol (MCP)** to ingest unstructured "dark data" without compromising privacy.

---

## 🏗️ Architecture & Agent Roles

Aegis operates as a cyclic graph. If the system’s internal confidence threshold is not met, the **Supervisor** triggers a refinement loop, forcing agents to re-evaluate their findings.

| Agent | Responsibility | Logic Layer |
| :--- | :--- | :--- |
| **Supervisor** | Orchestration & State Management | LangGraph |
| **Researcher** | Context Synthesis & Feature Extraction | MCP + Llama 3.2 |
| **Optimizer** | Predictive Risk Calculation | Scikit-Learn (Random Forest) |

### **The Predictive Model**
The Optimizer calculates the final risk probability using a weighted feature vector:
$$P(\text{Risk}) = \frac{\sum_{i=1}^{n} (w_i \cdot v_i)}{\text{Confidence}}$$
Where $w_i$ is the feature weight, $v_i$ is the extracted value, and the confidence is derived from the Researcher’s reliability scores.

---

## 🚀 Key Technical Features

* **Defensive Agentic Logic:** Implements a "Gatekeeper" pattern in the Researcher agent to identify data gaps and prevent LLM hallucinations.
* **A2A Refinement Loop:** The Supervisor can initiate up to 3 refinement cycles if data extraction is incomplete.
* **MCP Integration:** Secure local file ingestion using the **Model Context Protocol**, keeping sensitive financial reports off the public cloud.
* **ASGI Lifespan Management:** Optimized FastAPI startup that compiles the LangGraph mesh once, ensuring high-concurrency stability.
* **Python 3.12 Native:** Fully refactored to utilize stable `asyncio` loops and the latest `langchain-ollama` integration.

---

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Frameworks** | LangGraph, Google ADK, FastAPI |
| **Inference** | Ollama (Llama 3.2 3B) |
| **ML Engine** | Scikit-Learn, Joblib, Numpy |
| **Protocols** | MCP, WebSockets, GraphQL |

---

## ⚙️ Setup & Installation

### **1. Prerequisites**
* **Ollama** installed and running (`ollama run llama3.2`).
* **Python 3.12** (Stable release recommended).

### **2. Installation**
```powershell
# Clone the repository
git clone https://github.com/pradeepdotidl/aegis-risk-mesh.git
cd aegis-risk-mesh

# Create a stable virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### **3. Initialize the Mesh**
```powershell
# Train the local ML model
python ml_engine/train_baseline.py

# Start the MCP Server (Terminal 1)
python mcp_server/server.py

# Launch the API (Terminal 2)
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## 📊 Usage Example

### **POST /api/v1/analyze**
Trigger a multi-agent analysis for a specific entity.

**Request:**
```json
{
  "entity_name": "TCS",
  "analysis_timeframe": "30_days"
}
```

**Response:**
```json
{
  "entity_name": "TCS",
  "overall_risk_score": 14.5,
  "risk_category": "Safe",
  "researcher_summary": "Analysis verified after 1 cycles.",
  "refinement_cycles_used": 1
}
```

---

## 🤝 Contributing

This project is open to contributions. Since the mesh utilizes a strict A2A protocol, please ensure any new agents added to the graph include:
1.  Defined **State Schema** in `core/state.py`.
2.  **Defensive error handling** for null tool returns.
3.  Updated **unit tests** for ML feature consistency.

---

> **Note:** This project was developed as a proof-of-concept for secure, agentic financial intelligence. Always ensure `data/secure_docs` is excluded from version control to protect sensitive information.

---