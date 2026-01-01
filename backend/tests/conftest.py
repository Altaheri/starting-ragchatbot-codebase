"""
Shared fixtures for RAG chatbot tests

Uses lazy imports for heavy dependencies (vector_store, search_tools) to avoid
loading sentence_transformers when running lightweight tests like API tests.
"""
import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# API Testing Fixtures (no heavy dependencies)
# ---------------------------------------------------------------------------

def create_test_app(mock_rag_system):
    """
    Create a FastAPI test app without static file mounting.

    This avoids the issue where the main app.py mounts static files
    from a directory that doesn't exist in the test environment.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System - Test")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store reference to the mock RAG system
    app.state.rag_system = mock_rag_system

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        name: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG Chatbot API"}

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()

            answer, sources = rag.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag = app.state.rag_system
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            rag = app.state.rag_system
            rag.session_manager.delete_session(session_id)
            return {"status": "success", "message": "Session deleted"}
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

    return app


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = MagicMock()

    # Mock session manager
    mock_session_manager = MagicMock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_session_manager.delete_session.return_value = None
    mock_rag.session_manager = mock_session_manager

    # Mock query method
    mock_rag.query.return_value = (
        "This is a test answer about machine learning.",
        [{"name": "Test Course - Lesson 1", "url": "https://example.com/lesson/1"}]
    )

    # Mock get_course_analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["ML Basics", "Deep Learning", "NLP Fundamentals"]
    }

    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system"""
    return create_test_app(mock_rag_system)


@pytest.fixture
def test_client(test_app):
    """Create a test client for API testing"""
    from starlette.testclient import TestClient
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Vector Store and Search Tools Fixtures (lazy imports)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing"""
    from vector_store import VectorStore, SearchResults

    mock_store = MagicMock(spec=VectorStore)

    # Setup default behavior for search
    mock_store.search.return_value = SearchResults(
        documents=["This is test content about machine learning."],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )

    # Setup course resolution
    mock_store._resolve_course_name.return_value = "Test Course"

    # Setup lesson link retrieval
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"

    # Setup course catalog get
    mock_store.course_catalog = MagicMock()
    mock_store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'Test Course',
            'course_link': 'https://example.com/course',
            'lessons_json': json.dumps([
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Advanced Topics"}
            ])
        }],
        'ids': ['Test Course']
    }

    return mock_store


@pytest.fixture
def mock_empty_vector_store():
    """Create a mock VectorStore that returns empty results"""
    from vector_store import VectorStore, SearchResults

    mock_store = MagicMock(spec=VectorStore)

    # Setup empty search results
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

    # Course not found
    mock_store._resolve_course_name.return_value = None

    return mock_store


@pytest.fixture
def mock_error_vector_store():
    """Create a mock VectorStore that returns errors"""
    from vector_store import VectorStore, SearchResults

    mock_store = MagicMock(spec=VectorStore)

    # Setup error results
    mock_store.search.return_value = SearchResults.empty("Search error: Connection failed")

    return mock_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create CourseSearchTool with mock store"""
    from search_tools import CourseSearchTool
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create CourseOutlineTool with mock store"""
    from search_tools import CourseOutlineTool
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create ToolManager with registered tools"""
    from search_tools import ToolManager
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    from models import Course, Lesson
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/ml/1"),
            Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml/2"),
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    from models import CourseChunk
    return [
        CourseChunk(
            content="Machine learning is a subset of AI.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning uses labeled data.",
            course_title="Introduction to Machine Learning",
            lesson_number=2,
            chunk_index=1
        )
    ]


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return MockConfig()
