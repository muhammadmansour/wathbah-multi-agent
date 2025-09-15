#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2-friendly startup script for the Multi-Agent FastAPI system
"""
import uvicorn
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generaic_agent import app

if __name__ == "__main__":
    print("🚀 Starting Multi-Agent FastAPI Server with PM2...")
    print("📋 Available endpoints:")
    print("   • POST /ask - General agent endpoint")
    print("   • POST /multi-agent - Explicit multi-agent endpoint")
    print("   • GET /docs - API documentation")
    
    # Run with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Disable reload for PM2
        log_level="info"
    )
