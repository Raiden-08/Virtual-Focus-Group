"""
rag/explainer.py — Groq-powered Anomaly Explainer Agent
"""

import os
from groq import Groq

# Standard SWaT dataset sensor mapping (51 sensors)
SWAT_SENSORS = [
    "FIT101 (Raw Water Flow)", "LIT101 (Raw Water Level)", "MV101 (Raw Water Valve)",
    "P101 (Raw Water Pump 1)", "P102 (Raw Water Pump 2)", "AIT201 (NaCl level)",
    "AIT202 (HCl level)", "AIT203 (NaOCl level)", "FIT201 (Dosing Flow)",
    "MV201 (NaCl Valve)", "P201 (NaCl Pump)", "P202 (HCl Pump)",
    "P203 (NaOCl Pump)", "P204 (Dosing Pump Backup)", "P205 (Dosing Pump Backup)",
    "P206 (Dosing Pump Backup)", "DPIT301 (UF Feed Pressure)", "FIT301 (UF Feed Flow)",
    "LIT301 (UF Feed Level)", "MV301 (UF Feed Valve)", "MV302 (UF Valve 2)",
    "MV303 (UF Valve 3)", "MV304 (UF Valve 4)", "P301 (UF Feed Pump 1)",
    "P302 (UF Feed Pump 2)", "AIT401 (RO Feed ORP)", "AIT402 (RO Feed pH)",
    "FIT401 (RO Feed Flow)", "LIT401 (RO Feed Level)", "P401 (RO Feed Pump 1)",
    "P402 (RO Feed Pump 2)", "P403 (RO Feed Pump 3)", "P404 (RO Feed Pump 4)",
    "UV401 (RO UV Dechlorinator)", "AIT501 (RO Permeate pH)", "AIT502 (RO Permeate ORP)",
    "AIT503 (RO Permeate Conductivity)", "AIT504 (RO Permeate TOC)", "FIT501 (RO Permeate Flow)",
    "FIT502 (RO Reject Flow)", "FIT503 (RO Recirculation Flow)", "FIT504 (RO Product Flow)",
    "P501 (RO High Pressure Pump 1)", "P502 (RO High Pressure Pump 2)", "PIT501 (RO Feed Pressure)",
    "PIT502 (RO Permeate Pressure)", "PIT503 (RO Reject Pressure)", "FIT601 (UF Backwash Flow)",
    "P601 (UF Backwash Pump 1)", "P602 (UF Backwash Pump 2)", "P603 (UF Backwash Pump 3)"
]

def generate_incident_report(
    node_id: int, 
    reconstruction_error: float, 
    h_score: float, 
    h_temp: float, 
    h_struct: float, 
    h_rag: float, 
    neighbors: list
) -> str:
    """Queries Groq to generate a human-readable explanation of the anomaly."""
    
    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "ERROR: GROQ_API_KEY environment variable not set. Cannot generate report."
        
    client = Groq(api_key=api_key)
    
    sensor_name = SWAT_SENSORS[node_id] if node_id < len(SWAT_SENSORS) else f"Unknown Sensor {node_id}"
    
    # Format the neighbor data from the FAISS store
    if not neighbors:
        neighbor_context = "- No historical neighbors found."
    else:
        neighbor_context = "\n".join(
            [f"- Historical Case {i+1}: Known Anomaly={bool(n['label'])}, Vector Distance={n['dist']:.4f}" 
             for i, n in enumerate(neighbors)]
        )

    prompt = f"""
    You are an expert industrial control systems AI. An anomaly has just been detected in a water treatment plant.
    Analyze the telemetry data below and output a concise Incident Report.

    [SYSTEM TELEMETRY]
    - Triggering Sensor: {sensor_name}
    - Raw Reconstruction Error: {reconstruction_error:.4f} (High error = high deviation from normal)
    
    [RC-TGAD HARDNESS METRICS]
    - Total Hardness (H): {h_score:.4f} (0 to 1 scale. Higher means this is a highly subtle/complex anomaly)
    - Temporal Hardness (H_temp): {h_temp:.4f} (Subtlety of the signal deviation)
    - Structural Hardness (H_struct): {h_struct:.4f} (Graph network isolation)
    - Retrieval Entropy (H_RAG): {h_rag:.4f} (Uncertainty based on historical similarity)
    
    [RETRIEVED HISTORICAL NEIGHBORS (FAISS RAG)]
    {neighbor_context}

    Provide your response in the following format exactly. Keep it under 150 words total. Do not use markdown blocks or bold text, just plain text.
    
    LIKELY CAUSE: [Your brief interpretation of what physical event might cause this deviation on this specific sensor]
    SEVERITY: [Low/Medium/High/Critical based on the Hardness metrics and Error]
    RECOMMENDED ACTION: [One sentence operational recommendation]
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR GENERATING REPORT: {str(e)}"