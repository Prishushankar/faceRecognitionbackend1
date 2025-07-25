# --- Part 0: Imports (with new additions for self-pinging) ---
import os
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from url import compare_faces_from_urls  # Assuming this is your custom module

app = FastAPI(
    title="Face Comparison API",
    description="API for comparing faces in images using DeepFace",
    version="1.0.0",
)

# --- Part 1: Self-Pinging Logic ---
def ping_self():
    """
    Sends a GET request to the root URL of the app to keep it alive.
    """
    try:
        # Render provides the `RENDER_EXTERNAL_URL` environment variable
        app_url = os.environ.get("RENDER_EXTERNAL_URL")
        if app_url:
            print(f"Pinging {app_url} to keep alive...")
            requests.get(app_url, timeout=10)
            print("Ping successful.")
        else:
            print("RENDER_EXTERNAL_URL not set. Cannot ping self.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to ping self: {e}")

@app.on_event("startup")
def startup_event():
    """
    Initializes the scheduler and adds the ping job when the app starts.
    """
    scheduler = BackgroundScheduler()
    # Ping every 14 minutes
    scheduler.add_job(ping_self, 'interval', minutes=14)
    scheduler.start()
    print("Scheduler started. App will be pinged every 14 minutes.")
# --- End of Self-Pinging Logic ---

# Get allowed origins from environment variable or use default
# For production, you should specify your frontend domain
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", 
    "http://localhost:5173,http://localhost:5174,http://localhost:3000,https://your-frontend-domain.com"
).split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for initial testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompareRequest(BaseModel):
    urls: List[str]
    
@app.get("/")
def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok", "service": "face-comparison-api"}

@app.post("/compare")
def compare(request: CompareRequest):
    urls = request.urls
    print(f"Received URLs: {urls}")
    
    # Check if we have valid URLs
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        return {"error": "No valid URLs provided", "matrix": [], "distances": []}
    
    n = len(urls)
    results = []
    distances = []
    errors = []
    
    for i in range(n):
        row = []
        distance_row = []
        for j in range(n):
            if i == j:
                row.append(True)
                distance_row.append(0.0)
                print(f"Self comparison [{i}][{j}]: True, distance: 0.0")
            else:
                try:
                    # Special case - if URLs are identical, they must match
                    if urls[i] == urls[j] and urls[i].strip() != "":
                        print(f"Same URL detected for [{i}][{j}]: Auto-match")
                        row.append(True)
                        distance_row.append(0.0)
                    # If all URLs are the same non-empty URL, mark all as matches
                    elif all(url == urls[0] for url in urls if url.strip()) and urls[0].strip() != "" and urls[i].strip() != "" and urls[j].strip() != "":
                        print(f"All URLs are identical - forcing match for [{i}][{j}]")
                        row.append(True)
                        distance_row.append(0.0)
                    else:
                        res = compare_faces_from_urls(urls[i], urls[j])
                        verified = res["deepface_verified"]
                        distance = res["deepface_distance"]
                        cosine_sim = res.get("cosine_similarity", 0)
                        
                        # More debug information to help diagnose matching issues
                        print(f"Comparison [{i}][{j}]: verified={verified}, distance={distance:.4f}, cosine_sim={cosine_sim:.4f}")
                        
                        # Add the results to our matrices
                        row.append(verified)
                        distance_row.append(round(distance, 3))
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error comparing [{i}][{j}]: {error_msg}")
                    row.append(False)
                    distance_row.append(1.0)
                    errors.append(f"Error comparing image {i+1} with image {j+1}: {error_msg}")
        results.append(row)
        distances.append(distance_row)
    
    print(f"Final matrix: {results}")
    print(f"Distance matrix: {distances}")
    
    # Get the threshold from the comparison function (consistent across all comparisons)
    threshold = 0.25  # This matches the threshold used in url.py
    
    response = {
        "matrix": results, 
        "distances": distances,
        "threshold": threshold
    }
    
    # Include errors if any occurred
    if errors:
        response["errors"] = errors
        response["has_errors"] = True
        print(f"Returning with {len(errors)} errors")
    
    return response

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT if available (for cloud platforms) or default to 8001
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
