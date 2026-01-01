"""
Tests for RAG system handling content-related queries in rag_system.py

These tests evaluate:
1. End-to-end query handling
2. Tool integration
3. Source retrieval
4. Session management integration
5. Error scenarios that might cause 'query failed'
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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


class MockContentBlock:
    """Mock for Anthropic content block"""
    def __init__(self, block_type, text=None, name=None, input_data=None, block_id=None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input_data or {}
        self.id = block_id or "tool_use_123"


class MockResponse:
    """Mock for Anthropic API response"""
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class TestRAGSystemQuery:
    """Tests for RAG system query method"""

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query_returns_response_and_sources(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that query returns both response and sources"""
        # Setup mocks
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Direct response without tool use
        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="This is the answer")]
        )
        mock_client.messages.create.return_value = mock_response

        # Setup ChromaDB mock
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        response, sources = rag.query("What is machine learning?")

        assert response is not None
        assert isinstance(response, str)
        assert isinstance(sources, list)

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query_with_tool_use_executes_search(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that tool use triggers search and returns results"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # First response: tool use
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "machine learning basics"},
                    block_id="tool_123"
                )
            ]
        )

        # Second response: final answer
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Machine learning is...")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup ChromaDB mock with actual results
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [["Machine learning is a subset of AI"]],
            'metadatas': [[{"course_title": "ML Course", "lesson_number": 1}]],
            'distances': [[0.1]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        response, sources = rag.query("What is machine learning?")

        assert "Machine learning" in response
        assert mock_client.messages.create.call_count == 2

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query_returns_sources_after_search(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that sources are populated after tool search"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "test"},
                    block_id="tool_123"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [["Content"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
            'distances': [[0.1]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        # Mock get_lesson_link
        rag.vector_store.get_lesson_link = MagicMock(return_value="https://example.com/lesson")

        response, sources = rag.query("Test query")

        # Sources should be populated from the search
        assert len(sources) > 0
        assert "Test Course" in sources[0]["name"]

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query_with_session_includes_history(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that session history is passed to AI generator"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Follow-up answer")]
        )
        mock_client.messages.create.return_value = mock_response

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        # Create session and add history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "Previous Q", "Previous A")

        rag.query("Follow-up question", session_id=session_id)

        # Verify the API was called with history in system prompt
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous Q" in call_kwargs["system"] or "Previous A" in call_kwargs["system"]


class TestRAGSystemToolRegistration:
    """Tests for tool registration in RAG system"""

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_both_tools_are_registered(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that both search and outline tools are registered"""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        tool_definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_definitions]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_tool_manager_can_execute_search_tool(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that tool manager can execute search tool"""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [["Test content"]],
            'metadatas': [[{"course_title": "Test", "lesson_number": 1}]],
            'distances': [[0.1]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        # Mock get_lesson_link
        rag.vector_store.get_lesson_link = MagicMock(return_value=None)

        result = rag.tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        assert result is not None
        assert isinstance(result, str)

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_tool_manager_can_execute_outline_tool(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that tool manager can execute outline tool"""
        import json

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {
            'ids': ['Test Course'],
            'metadatas': [{
                'title': 'Test Course',
                'course_link': 'https://example.com',
                'lessons_json': json.dumps([
                    {"lesson_number": 1, "lesson_title": "Intro"}
                ])
            }]
        }
        mock_collection.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{"title": "Test Course"}]],
            'distances': [[0.1]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        result = rag.tool_manager.execute_tool(
            "get_course_outline",
            course_title="Test Course"
        )

        assert "Test Course" in result
        assert "Lesson 1" in result


class TestRAGSystemErrorScenarios:
    """Tests for error scenarios that might cause 'query failed'"""

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_handles_empty_search_results(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test handling when search returns no results"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "nonexistent topic"},
                    block_id="tool_123"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="No results found")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        response, sources = rag.query("Nonexistent topic")

        # Should not crash, should return appropriate message
        assert response is not None
        assert "failed" not in response.lower() or "no" in response.lower()

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_handles_api_error_gracefully(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test handling of API errors"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Simulate API error
        mock_client.messages.create.side_effect = Exception("API Error")

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        # Should raise or handle gracefully
        with pytest.raises(Exception):
            rag.query("Test query")

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_handles_chromadb_search_error(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test handling of ChromaDB search errors"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "test"},
                    block_id="tool_123"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Error occurred")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        # Simulate ChromaDB error
        mock_collection.query.side_effect = Exception("ChromaDB Error")

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)

        # The tool should catch the error and return error message
        response, sources = rag.query("Test query")

        # Should contain error indication
        assert response is not None


class TestRAGSystemSourcesHandling:
    """Tests specifically for sources handling"""

    @patch('anthropic.Anthropic')
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_sources_are_reset_after_query(self, mock_embedding, mock_chroma, mock_anthropic):
        """Test that sources are reset between queries"""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "test"},
                    block_id="tool_123"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )

        # Direct response for second query
        direct_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Direct answer")]
        )

        # Provide enough responses for both queries
        mock_client.messages.create.side_effect = [
            tool_use_response, final_response,  # First query (tool use + final)
            direct_response  # Second query (direct)
        ]

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}
        mock_collection.query.return_value = {
            'documents': [["Content"]],
            'metadatas': [[{"course_title": "Test", "lesson_number": 1}]],
            'distances': [[0.1]]
        }

        from rag_system import RAGSystem
        config = MockConfig()
        rag = RAGSystem(config)
        rag.vector_store.get_lesson_link = MagicMock(return_value=None)

        # First query - with tool use, should have sources
        response1, sources1 = rag.query("First query")
        assert len(sources1) > 0  # Should have sources from search

        # Second query - direct response without search
        response2, sources2 = rag.query("General question")

        # Sources should be empty for second query since sources were reset
        # after first query and no new search was performed
        assert sources2 == []  # Sources should be empty after reset
