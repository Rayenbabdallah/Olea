# ═══════════════════════════════════════════════════════════════════════
#  OLEA AI — Production Dockerfile (Backend + Frontend)
# ═══════════════════════════════════════════════════════════════════════

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production image
FROM python:3.10-slim
WORKDIR /app

# Install backend dependencies (cached layer)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY train.csv ./train.csv

# Copy built frontend into backend/static
COPY --from=frontend-build /app/frontend/dist ./backend/static

# Expose port (Render uses $PORT)
ENV PORT=8000
EXPOSE 8000

# Start server
CMD ["sh", "-c", "cd backend && python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
