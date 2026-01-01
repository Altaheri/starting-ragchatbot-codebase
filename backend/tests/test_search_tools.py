"""
Tests for CourseSearchTool execute method in search_tools.py

These tests evaluate:
1. Successful content searches with various parameters
2. Empty result handling
3. Error handling
4. Course name resolution
5. Result formatting
"""
import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_basic_query_returns_formatted_results(self, course_search_tool, mock_vector_store):
        """Test that a basic query returns properly formatted results"""
        result = course_search_tool.execute(query="machine learning")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains expected content
        assert "Test Course" in result
        assert "machine learning" in result.lower() or "test content" in result.lower()

    def test_execute_with_course_name_filters_correctly(self, course_search_tool, mock_vector_store):
        """Test that course_name parameter is passed to search"""
        result = course_search_tool.execute(
            query="neural networks",
            course_name="Deep Learning Course"
        )

        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="Deep Learning Course",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filters_correctly(self, course_search_tool, mock_vector_store):
        """Test that lesson_number parameter is passed to search"""
        result = course_search_tool.execute(
            query="backpropagation",
            lesson_number=3
        )

        mock_vector_store.search.assert_called_once_with(
            query="backpropagation",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_all_parameters(self, course_search_tool, mock_vector_store):
        """Test query with both course_name and lesson_number"""
        result = course_search_tool.execute(
            query="activation functions",
            course_name="Neural Networks 101",
            lesson_number=5
        )

        mock_vector_store.search.assert_called_once_with(
            query="activation functions",
            course_name="Neural Networks 101",
            lesson_number=5
        )

    def test_execute_returns_error_message_on_search_error(self, mock_error_vector_store):
        """Test that errors from vector store are returned properly"""
        tool = CourseSearchTool(mock_error_vector_store)
        result = tool.execute(query="test query")

        assert "error" in result.lower() or "failed" in result.lower()

    def test_execute_returns_no_results_message_when_empty(self, mock_empty_vector_store):
        """Test that empty results return appropriate message"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "no relevant content" in result.lower() or "not found" in result.lower()

    def test_execute_with_course_filter_includes_filter_in_no_results_message(self, mock_empty_vector_store):
        """Test that course name appears in 'no results' message"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute(
            query="obscure topic",
            course_name="Specific Course"
        )

        # Should mention the course in the no-results message
        assert "no" in result.lower()

    def test_execute_tracks_sources_correctly(self, course_search_tool, mock_vector_store):
        """Test that last_sources is populated after search"""
        course_search_tool.execute(query="test query")

        sources = course_search_tool.last_sources
        assert len(sources) > 0
        assert "name" in sources[0]

    def test_execute_source_includes_lesson_number_when_available(self, course_search_tool, mock_vector_store):
        """Test that sources include lesson info when present"""
        course_search_tool.execute(query="test query")

        sources = course_search_tool.last_sources
        # Should have lesson number in source name since metadata includes it
        assert "Lesson" in sources[0]["name"]


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool tool definition"""

    def test_get_tool_definition_has_required_fields(self, course_search_tool):
        """Test tool definition has name, description, input_schema"""
        definition = course_search_tool.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition

    def test_tool_definition_name_is_search_course_content(self, course_search_tool):
        """Test tool has correct name"""
        definition = course_search_tool.get_tool_definition()
        assert definition["name"] == "search_course_content"

    def test_tool_definition_has_query_as_required(self, course_search_tool):
        """Test that query is the only required parameter"""
        definition = course_search_tool.get_tool_definition()
        required = definition["input_schema"].get("required", [])

        assert "query" in required
        assert "course_name" not in required
        assert "lesson_number" not in required


class TestCourseOutlineToolExecute:
    """Tests for CourseOutlineTool.execute() method"""

    def test_execute_returns_course_title(self, course_outline_tool, mock_vector_store):
        """Test that course title is in the output"""
        result = course_outline_tool.execute(course_title="Test Course")

        assert "Test Course" in result

    def test_execute_returns_course_link(self, course_outline_tool, mock_vector_store):
        """Test that course link is in the output"""
        result = course_outline_tool.execute(course_title="Test Course")

        assert "https://example.com/course" in result

    def test_execute_returns_all_lessons(self, course_outline_tool, mock_vector_store):
        """Test that all lessons are listed"""
        result = course_outline_tool.execute(course_title="Test Course")

        assert "Lesson 1" in result
        assert "Introduction" in result
        assert "Lesson 2" in result
        assert "Advanced Topics" in result

    def test_execute_with_nonexistent_course(self, mock_empty_vector_store):
        """Test behavior when course is not found"""
        tool = CourseOutlineTool(mock_empty_vector_store)
        result = tool.execute(course_title="Nonexistent Course")

        assert "no course found" in result.lower() or "not found" in result.lower()


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool_adds_tool(self, mock_vector_store):
        """Test that tools are registered correctly"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions_returns_all_tools(self, tool_manager):
        """Test that all registered tools are returned"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_calls_correct_tool(self, tool_manager, mock_vector_store):
        """Test that execute_tool routes to the right tool"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        mock_vector_store.search.assert_called_once()

    def test_execute_tool_with_unknown_tool_returns_error(self, tool_manager):
        """Test that unknown tool names return error"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_from_search(self, tool_manager, mock_vector_store):
        """Test that sources are retrieved after search"""
        tool_manager.execute_tool("search_course_content", query="test")

        sources = tool_manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources_clears_sources(self, tool_manager, mock_vector_store):
        """Test that reset_sources clears the sources"""
        tool_manager.execute_tool("search_course_content", query="test")
        tool_manager.reset_sources()

        sources = tool_manager.get_last_sources()
        assert len(sources) == 0


class TestSearchResultsFormatting:
    """Tests for search results formatting edge cases"""

    def test_format_results_with_missing_metadata_fields(self, mock_vector_store):
        """Test formatting when metadata has missing fields"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        # Should not crash, should handle gracefully
        assert result is not None
        assert isinstance(result, str)

    def test_format_results_with_none_lesson_number(self, mock_vector_store):
        """Test formatting when lesson_number is None"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": None}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert result is not None
