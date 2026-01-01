"""
Tests for FastAPI endpoints in the RAG chatbot system.

These tests use a test app defined in conftest.py that mirrors the API
endpoints from app.py but without static file mounting (which requires
files that don't exist in the test environment).

Tests cover:
1. Root endpoint
2. POST /api/query - Query processing
3. GET /api/courses - Course statistics
4. DELETE /api/sessions/{session_id} - Session management
5. Error handling scenarios
"""
import pytest
from unittest.mock import MagicMock


class TestRootEndpoint:
    """Tests for the root endpoint"""

    def test_root_returns_ok_status(self, test_client):
        """Test that root endpoint returns success status"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_message(self, test_client):
        """Test that root endpoint returns API message"""
        response = test_client.get("/")

        data = response.json()
        assert "message" in data


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_with_valid_request_returns_200(self, test_client):
        """Test that valid query returns 200 status"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200

    def test_query_returns_answer_field(self, test_client):
        """Test that response contains answer field"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_returns_sources_field(self, test_client):
        """Test that response contains sources field"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_returns_session_id(self, test_client):
        """Test that response contains session_id field"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_query_creates_new_session_when_not_provided(self, test_client, mock_rag_system):
        """Test that new session is created when session_id not provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "test query"}
        )

        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, test_client, mock_rag_system):
        """Test that provided session_id is used"""
        response = test_client.post(
            "/api/query",
            json={"query": "test query", "session_id": "existing-session"}
        )

        # Verify query was called with the provided session_id
        mock_rag_system.query.assert_called_with("test query", "existing-session")

    def test_query_calls_rag_system_query(self, test_client, mock_rag_system):
        """Test that RAG system query method is called"""
        test_client.post(
            "/api/query",
            json={"query": "What is deep learning?"}
        )

        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == "What is deep learning?"

    def test_query_source_has_name_field(self, test_client):
        """Test that sources contain name field"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        data = response.json()
        if len(data["sources"]) > 0:
            assert "name" in data["sources"][0]

    def test_query_source_has_url_field(self, test_client):
        """Test that sources contain url field"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        data = response.json()
        if len(data["sources"]) > 0:
            assert "url" in data["sources"][0]

    def test_query_without_query_field_returns_422(self, test_client):
        """Test that missing query field returns validation error"""
        response = test_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422

    def test_query_with_empty_query_is_accepted(self, test_client):
        """Test that empty query string is accepted (validation at RAG level)"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string is valid JSON, RAG system handles validation
        assert response.status_code == 200

    def test_query_error_returns_500(self, test_client, mock_rag_system):
        """Test that RAG system errors return 500 status"""
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = test_client.post(
            "/api/query",
            json={"query": "test query"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_200(self, test_client):
        """Test that courses endpoint returns 200 status"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200

    def test_courses_returns_total_courses(self, test_client):
        """Test that response contains total_courses field"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert "total_courses" in data
        assert isinstance(data["total_courses"], int)

    def test_courses_returns_course_titles(self, test_client):
        """Test that response contains course_titles field"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)

    def test_courses_returns_correct_count(self, test_client):
        """Test that total_courses matches number of titles"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3

    def test_courses_returns_expected_titles(self, test_client):
        """Test that course titles match expected values"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert "ML Basics" in data["course_titles"]
        assert "Deep Learning" in data["course_titles"]
        assert "NLP Fundamentals" in data["course_titles"]

    def test_courses_calls_get_course_analytics(self, test_client, mock_rag_system):
        """Test that get_course_analytics is called"""
        test_client.get("/api/courses")

        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_error_returns_500(self, test_client, mock_rag_system):
        """Test that analytics errors return 500 status"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestSessionsEndpoint:
    """Tests for DELETE /api/sessions/{session_id} endpoint"""

    def test_delete_session_returns_success(self, test_client):
        """Test that valid session deletion returns success"""
        response = test_client.delete("/api/sessions/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_delete_session_returns_message(self, test_client):
        """Test that deletion returns confirmation message"""
        response = test_client.delete("/api/sessions/test-session-123")

        data = response.json()
        assert "message" in data
        assert "deleted" in data["message"].lower()

    def test_delete_session_calls_session_manager(self, test_client, mock_rag_system):
        """Test that session_manager.delete_session is called"""
        test_client.delete("/api/sessions/my-session-id")

        mock_rag_system.session_manager.delete_session.assert_called_once_with("my-session-id")

    def test_delete_nonexistent_session_returns_404(self, test_client, mock_rag_system):
        """Test that deleting nonexistent session returns 404"""
        mock_rag_system.session_manager.delete_session.side_effect = KeyError("Session not found")

        response = test_client.delete("/api/sessions/nonexistent-session")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestRequestValidation:
    """Tests for request validation across endpoints"""

    def test_query_with_invalid_json_returns_422(self, test_client):
        """Test that invalid JSON returns validation error"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_with_wrong_type_returns_422(self, test_client):
        """Test that wrong field type returns validation error"""
        response = test_client.post(
            "/api/query",
            json={"query": 12345}  # Should be string
        )

        assert response.status_code == 422

    def test_query_accepts_null_session_id(self, test_client):
        """Test that null session_id is accepted"""
        response = test_client.post(
            "/api/query",
            json={"query": "test", "session_id": None}
        )

        assert response.status_code == 200


class TestResponseFormat:
    """Tests for response format consistency"""

    def test_query_response_has_correct_content_type(self, test_client):
        """Test that query response has JSON content type"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        assert "application/json" in response.headers["content-type"]

    def test_courses_response_has_correct_content_type(self, test_client):
        """Test that courses response has JSON content type"""
        response = test_client.get("/api/courses")

        assert "application/json" in response.headers["content-type"]

    def test_error_response_has_detail_field(self, test_client, mock_rag_system):
        """Test that error responses include detail field"""
        mock_rag_system.query.side_effect = Exception("Test error")

        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        data = response.json()
        assert "detail" in data
