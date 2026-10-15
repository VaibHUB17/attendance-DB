# Employee Attendance & Analytics API

## How to Run

1. Configure environment variables (`MONGO_URI` and `MONGO_DB`, or set them in `.env`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   uvicorn app.main:app --port 8000
   ```

## Requirements & Setup
- **MongoDB 6.0 or newer** is required (`$densify`, and `$unionWith` with `$documents`).
- All indexes are created at startup.
