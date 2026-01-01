"""
Shared fixtures for RAG chatbot tests
"""
import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing"""
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
    mock_store = MagicMock(spec=VectorStore)

    # Setup error results
    mock_store.search.return_value = SearchResults.empty("Search error: Connection failed")

    return mock_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create CourseSearchTool with mock store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create CourseOutlineTool with mock store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
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
