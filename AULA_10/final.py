import nest_asyncio
from pyngrok import ngrok
import uvicorn
import threading
import time
import requests

nest_asyncio.apply()

# Set your ngrok authtoken here. Replace 'YOUR_AUTHTOKEN' with your actual token.
# You can get a token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token('') # Using the token from cell jEIG2wHBwzrM

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8001) # Changed port to 8001

# Start uvicorn in a separate thread
uvicorn_thread = threading.Thread(target=run_uvicorn)
uvicorn_thread.daemon = True # Allows the main program to exit even if the thread is still running
uvicorn_thread.start()

# Give uvicorn some time to start
time.sleep(2)

# Establish ngrok tunnel
try:
    public_url = ngrok.connect(8001) # Changed port to 8001
    print("URL pública do ngrok:", public_url)
except Exception as e:
    print(f"Erro ao conectar ngrok: {e}")
    # In case ngrok connection fails, we can't proceed with the request
    exit()

# Construct the full API endpoint URL by accessing the public_url attribute of the NgrokTunnel object
api_url = f"{public_url.public_url}/predict"

# Make the POST request to the API
response = requests.post(api_url, params={
 	"idade": 40,
 	"salario": 7000,
 	"tempo_cliente": 5
 })

print(response.json())

# It's good practice to disconnect ngrok when you're done, though for Colab it might be automatically handled
# ngrok.disconnect(public_url)
