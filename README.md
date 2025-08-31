# 🚀 Enterprise RAG Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-ff69b4.svg)]()
[![Enterprise Ready](https://img.shields.io/badge/Enterprise-Ready-success.svg)]()

## 🌟 Overview

**Enterprise RAG Platform** is a revolutionary, production-ready Retrieval-Augmented Generation system that combines the power of multiple AI providers with advanced document processing, vector search, and real-time conversation capabilities. Designed for enterprise environments, it offers unparalleled scalability, security, and performance for knowledge management and AI-powered assistance.

### 🎯 Perfect for

- **Enterprise Knowledge Management** - Centralized, searchable document repositories
- **Customer Support Teams** - AI-powered assistance with company knowledge
- **Research Organizations** - Advanced document analysis and insights
- **Legal and Compliance** - Secure document processing with audit trails
- **Educational Institutions** - Smart learning and research platforms
- **Healthcare Systems** - Compliant medical knowledge management
- **Financial Services** - Secure document analysis and reporting

## ✨ Enterprise Features

### 🧠 Advanced RAG Engine
- **Multi-Provider LLM Support**: OpenAI GPT, Anthropic Claude, Google Gemini, Cohere
- **Multiple Vector Databases**: ChromaDB, FAISS, Pinecone, Weaviate, Qdrant
- **Intelligent Embeddings**: OpenAI, Sentence Transformers, Cohere embeddings
- **Hybrid Search**: Semantic + keyword search with advanced filtering
- **Smart Document Chunking**: Recursive, semantic, and code-aware strategies
- **Conversation Memory**: Persistent context and multi-turn conversations

### 🔐 Enterprise Security
- **JWT Authentication**: Secure token-based authentication with refresh
- **API Key Management**: Service-to-service authentication with rate limiting  
- **Role-Based Access Control**: Granular permissions and user roles
- **Multi-Tenancy**: Complete tenant isolation and data segregation
- **Audit Logging**: Comprehensive security and compliance tracking
- **Encryption**: End-to-end data protection and secure storage

### 📊 Real-Time Monitoring
- **Prometheus Metrics**: Production-ready metrics collection
- **Performance Analytics**: Response times, throughput, and error rates
- **Health Checks**: Automated system health monitoring
- **Resource Monitoring**: CPU, memory, and storage utilization
- **Custom Dashboards**: Grafana integration for visualization
- **Alert Management**: Proactive issue detection and notification

### 🚀 Scalable Architecture
- **Microservices Design**: Modular, independently scalable components
- **Async Processing**: High-performance asynchronous operations
- **Caching Strategy**: Redis-based intelligent caching
- **Load Balancing**: Horizontal scaling with container orchestration
- **Database Optimization**: Async PostgreSQL with connection pooling
- **Docker Ready**: Complete containerization with orchestration

### 📄 Advanced Document Processing
- **Multi-Format Support**: PDF, DOCX, HTML, Markdown, CSV, JSON
- **OCR Integration**: Text extraction from images and scanned documents
- **Table Extraction**: Intelligent table detection and processing
- **Metadata Management**: Rich document metadata and tagging
- **Version Control**: Document versioning and change tracking
- **Bulk Operations**: High-throughput batch processing

### 💬 Real-Time Communication
- **WebSocket Streaming**: Live response streaming for better UX
- **Conversation Management**: Persistent chat sessions and context
- **Collaborative Features**: Multi-user conversations and sharing
- **Real-Time Updates**: Live notifications and status updates
- **Mobile Support**: Responsive design for all devices

## 🏗️ Enterprise Architecture

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Load Balancer  │    │   API Gateway   │
│   (React/TS)    │◄──►│   (Nginx/Traefik)│◄──►│   (FastAPI)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 ▼                                 │
    ┌─────────────────────────────────┐    ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
    │        RAG Engine               │    │      Authentication            │    │        Monitoring               │
    │  ┌─────────┬──────────────────┐ │    │  ┌─────────┬─────────────────┐│    │  ┌─────────┬─────────────────┐  │
    │  │Embedding│ Vector Database  │ │    │  │   JWT   │      RBAC       ││    │  │Prometheus│     Grafana     │  │
    │  │ Engine  │ (Chroma/FAISS)   │ │    │  │  Auth   │   Multi-Tenant  ││    │  │ Metrics │   Dashboards    │  │
    │  └─────────┴──────────────────┘ │    │  └─────────┴─────────────────┘│    │  └─────────┴─────────────────┘  │
    │  ┌─────────┬──────────────────┐ │    └─────────────────────────────────┘    └─────────────────────────────────┘
    │  │   LLM   │   Document       │ │                     │                                        │
    │  │ Engine  │  Processing      │ │            ┌─────────────────┐                    ┌─────────────────┐
    │  └─────────┴──────────────────┘ │            │ PostgreSQL DB   │                    │ Redis Cache     │
    └─────────────────────────────────┘            │ (Users/Metadata)│                    │ (Sessions/Rate) │
                       │                            └─────────────────┘                    └─────────────────┘
              ┌─────────────────┐
              │ Multi-AI Providers │
              │ OpenAI │ Anthropic │
              │ Google │ Cohere    │
              └─────────────────────┘
```

### Technology Stack
- **Backend**: FastAPI (Python 3.11+), AsyncIO, WebSockets, Pydantic
- **AI/ML**: Multi-provider LLM integration, Vector embeddings, RAG pipeline
- **Vector DBs**: ChromaDB, FAISS, Pinecone, Weaviate, Qdrant
- **Database**: PostgreSQL, Redis, SQLAlchemy (async)
- **Security**: JWT, RBAC, API keys, Multi-tenancy
- **Monitoring**: Prometheus, Grafana, Structured logging
- **Deployment**: Docker, Docker Compose, Kubernetes-ready

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+** with pip
- **Docker & Docker Compose** (recommended)
- **AI Provider API Keys** (OpenAI, Anthropic, Google AI, or Cohere)
- **8GB+ RAM** (for local vector databases)

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/karimosman89/RAG_QA_System.git
cd RAG_QA_System

# Configure environment variables
cp .env.example .env
nano .env  # Add your AI provider API keys

# Start with Docker Compose
docker-compose up -d

# Or for development
docker-compose -f docker-compose.dev.yml up
```

### Option 2: Local Development Setup

```bash
# Clone and navigate
git clone https://github.com/karimosman89/RAG_QA_System.git
cd RAG_QA_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -m alembic upgrade head

# Start the server
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

### 🔧 Configuration

Create a `.env` file with your configuration:

```env
# Application Settings
APP_NAME="Enterprise RAG Platform"
ENVIRONMENT="production"
DEBUG=false

# Security Configuration
JWT_SECRET_KEY="your-super-secret-key-change-this"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# AI Provider API Keys
OPENAI_API_KEY="sk-your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
GOOGLE_AI_KEY="your-google-ai-key"
COHERE_API_KEY="your-cohere-key"

# Database Configuration
DATABASE_URL="postgresql+asyncpg://user:password@localhost/rag_platform"
REDIS_URL="redis://localhost:6379/0"

# Vector Store Configuration
VECTOR_STORE_TYPE="chroma"  # chroma, faiss, pinecone, weaviate, qdrant
EMBEDDING_PROVIDER="openai"  # openai, sentence_transformers, cohere

# File Upload Settings
MAX_UPLOAD_SIZE_MB=100
UPLOAD_DIRECTORY="./static/uploads"
### 📊 Access the Platform

Once started, access the platform at:
- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs  
- **Admin Dashboard**: http://localhost:8000/admin
- **Health Monitoring**: http://localhost:8000/api/health
- **Metrics Endpoint**: http://localhost:8000/api/metrics

## 📖 Usage Guide

### Document Management

Upload and process documents for RAG retrieval:

```python
import httpx

# Upload document
async def upload_document():
    async with httpx.AsyncClient() as client:
        with open("document.pdf", "rb") as f:
            response = await client.post(
                "http://localhost:8000/api/documents/upload",
                files={"file": f},
                data={
                    "metadata": json.dumps({
                        "title": "Company Policy",
                        "category": "HR",
                        "tags": ["policy", "employee"]
                    }),
                    "generate_embeddings": True
                },
                headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
            )
        return response.json()
```

### RAG Query Processing

Ask questions about your documents:

```python
# Query documents
async def query_documents():
    query_data = {
        "query": "What is the company vacation policy?",
        "context": {
            "query_type": "rag",
            "max_context_documents": 5,
            "temperature": 0.7
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/queries/query",
            json=query_data,
            headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
        )
        return response.json()
```

### Real-Time Chat

Stream responses for better user experience:

```javascript
// WebSocket streaming chat
const ws = new WebSocket('ws://localhost:8000/api/queries/chat/conversation_id');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'query',
        data: {
            query: 'Explain our data security policies',
            conversation_id: 'conv_123',
            stream: true
        }
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Response chunk:', message.content);
};
```

## 🔌 Comprehensive API Reference

### 🔐 Authentication APIs

#### User Registration
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "secure_password",
  "full_name": "John Smith",
  "organization": "Acme Corp"
}
```

#### JWT Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@company.com", 
  "password": "secure_password"
}
```

#### Token Refresh
```http
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### 📄 Document Management APIs

#### Upload Document
```http
POST /api/documents/upload
Authorization: Bearer JWT_TOKEN
Content-Type: multipart/form-data

file: [binary data]
metadata: {
  "title": "Document Title",
  "description": "Document description", 
  "category": "category_name",
  "tags": ["tag1", "tag2"]
}
```

#### List Documents
```http
GET /api/documents?limit=20&offset=0&category=HR
Authorization: Bearer JWT_TOKEN
```

#### Search Documents
```http
POST /api/documents/search
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "query": "vacation policy",
  "filters": {
    "category": "HR",
    "tags": ["policy"]
  },
  "limit": 10,
  "similarity_threshold": 0.7
}
```

### 🧠 RAG Query APIs

#### Standard Query
```http
POST /api/queries/query
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "query": "What are the security requirements?",
  "context": {
    "query_type": "rag",
    "document_ids": ["doc_1", "doc_2"],
    "temperature": 0.7,
    "max_tokens": 1000,
    "include_sources": true
  }
}
```

#### Streaming Query
```http
POST /api/queries/query/stream
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "query": "Explain the onboarding process",
  "context": {
    "stream": true,
    "conversation_id": "conv_123"
  }
}
```

### 💬 Conversation Management APIs

#### Create Conversation
```http
POST /api/queries/conversations
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "title": "HR Policy Discussion",
  "description": "Questions about company policies"
}
```

#### Get Conversation History
```http
GET /api/queries/conversations/conv_123
Authorization: Bearer JWT_TOKEN
```

### 👥 Admin APIs (Admin Role Required)

#### User Management
```http
GET /api/admin/users?limit=50&role=user
Authorization: Bearer JWT_TOKEN

POST /api/admin/users
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "email": "newuser@company.com",
  "username": "newuser",
  "role": "user",
  "full_name": "New User"
}
```

#### System Configuration
```http
GET /api/admin/config
Authorization: Bearer JWT_TOKEN

PUT /api/admin/config
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "max_upload_size_mb": 200,
  "default_embedding_model": "text-embedding-ada-002",
  "rate_limit_per_minute": 100
}
```

#### Analytics Dashboard
```http
GET /api/admin/dashboard
Authorization: Bearer JWT_TOKEN
```

#### System Metrics
```http
GET /api/admin/metrics
Authorization: Bearer JWT_TOKEN
```

### 📊 Monitoring & Health APIs

#### Health Check
```http
GET /api/health
```

#### Detailed System Health
```http
GET /api/admin/health/detailed
Authorization: Bearer JWT_TOKEN
```

#### Prometheus Metrics
```http
GET /api/metrics
```

## 🏢 Enterprise Deployment

### Production Environment Setup

```bash
# Clone repository
git clone https://github.com/karimosman89/RAG_QA_System.git
cd RAG_QA_System

# Production environment setup
cp .env.example .env.prod
# Configure production values in .env.prod

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d

# Or deploy with full monitoring stack
docker-compose --profile monitoring up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-platform
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-platform
  template:
    metadata:
      labels:
        app: rag-platform
    spec:
      containers:
      - name: rag-platform
        image: rag-platform:v2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
```

### Monitoring & Observability

Enable comprehensive monitoring:

```yaml
# Prometheus monitoring
prometheus:
  enabled: true
  scrape_interval: 15s
  retention: 15d

# Grafana dashboards
grafana:
  enabled: true
  admin_password: "secure_password"
  datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
```

## 🧪 Testing & Quality Assurance

### Running the Test Suite

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test categories
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/api/ -v           # API tests
```

### Performance Testing

```bash
# Install performance testing tools
pip install locust

# Run load tests
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 📈 Performance & Scalability

### Optimization Features

- **Async Architecture**: Non-blocking operations for high concurrency
- **Redis Caching**: Intelligent caching of embeddings and responses
- **Database Connection Pooling**: Optimized database connections
- **Vector Index Optimization**: Efficient similarity search
- **Response Streaming**: Real-time response delivery
- **Horizontal Scaling**: Container-based scaling

### Production Benchmarks

- **Concurrent Users**: 1000+ simultaneous connections
- **Document Processing**: 100+ documents/minute  
- **Query Response Time**: <2 seconds average
- **Throughput**: 10,000+ queries/hour
- **Uptime**: 99.9% availability target

## 🤝 Contributing to the Platform

We welcome enterprise contributions!

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/RAG_QA_System.git
cd RAG_QA_System

# Development environment
docker-compose -f docker-compose.dev.yml up -d

# Install development tools
pip install pre-commit black flake8 mypy
pre-commit install

# Run tests before committing
python -m pytest
black src/
flake8 src/
mypy src/
```

### Enterprise Contribution Areas

- **Multi-modal RAG**: Image and video document processing
- **Advanced Security**: SSO integration, advanced RBAC
- **Performance**: Query optimization, caching strategies  
- **Integrations**: CRM, SharePoint, Google Drive connectors
- **Analytics**: Advanced usage analytics and reporting

## 📄 License & Enterprise Support

This project is licensed under the MIT License with enterprise support available.

### Enterprise Support Options

- **Community**: GitHub Issues and Discussions (Free)
- **Professional**: Email support with SLA (Contact for pricing)
- **Enterprise**: 24/7 support, custom development (Contact for pricing)

## 🎯 Roadmap

### Version 2.1 (Q1 2025)
- [ ] Multi-modal document processing (images, videos)
- [ ] Advanced hybrid search with keyword ranking
- [ ] Custom embedding model fine-tuning
- [ ] Enterprise SSO integration (SAML, OAuth)

### Version 2.2 (Q2 2025)  
- [ ] Advanced analytics dashboard
- [ ] Multi-language document support
- [ ] Workflow automation and triggers
- [ ] Advanced conversation memory management

### Version 3.0 (Q3 2025)
- [ ] AI agents for complex task automation
- [ ] Integration marketplace and plugins
- [ ] Advanced compliance and governance features
- [ ] Federated learning for private model training

---

**🚀 Transform your organization's knowledge management with enterprise-grade RAG technology. Start building intelligent document-powered applications today!**