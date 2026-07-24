#!/bin/sh
# Backend runs quietly in the background on a fixed internal port
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 &

# Frontend runs in front, on the port Railway gives us
cd ../frontend && streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0