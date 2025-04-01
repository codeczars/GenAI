# Refactor Royale: AI Code Optimization Game - Setup Guide

## Prerequisites
Ensure you have the following installed:

1. **Ollama** - Download and install from [Ollama's official site](https://ollama.ai/).
2. **Mistral Model** - Install it using the command:
   ```sh
   ollama pull mistral
   ```
3. **Python** (latest version recommended)
4. **Node.js & npm** (latest version recommended)

---

## Backend Setup
1. Open a command prompt (CMD) and run the following command to start Ollama:
   ```sh
   ollama serve
   ```
2. Navigate to the `Backend` folder:
   ```sh
   cd backend
   ```
3. Open `server.py` in a code editor.
4. Run the backend server using:
   ```sh
   python server.py
   ```

---

## Frontend Setup
1. Navigate to the `Frontend` folder:
   ```sh
   cd frontend
   ```
2. Open the `src` directory and locate `App.js`.
3. Install dependencies by running:
   ```sh
   npm install
   ```
4. Start the frontend application using:
   ```sh
   npm start
   ```


---

## Notes
- Make sure **Ollama is running** before starting the backend.
- If you face any issues, check the logs in `server.py` and fix any dependency errors.
- You can modify frontend styles and logic inside the `src` folder.

