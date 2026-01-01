"""
Tests for AIGenerator tool calling in ai_generator.py

These tests evaluate:
1. Whether AIGenerator correctly invokes tools
2. Tool execution flow
3. Response handling after tool execution
4. Error handling during tool execution
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class MockContentBlock:
    """Mock for Anthropic content block"""

    def __init__(
        self, block_type, text=None, name=None, input_data=None, block_id=None
    ):
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


@pytest.fixture
def ai_generator():
    """Create AIGenerator with test API key"""
    return AIGenerator(api_key="test-api-key", model="claude-sonnet-4-20250514")


@pytest.fixture
def mock_tool_manager(mock_vector_store):
    """Create mock tool manager"""
    manager = ToolManager()
    tool = CourseSearchTool(mock_vector_store)
    manager.register_tool(tool)
    return manager


class TestAIGeneratorToolCalling:
    """Tests for AIGenerator tool calling behavior"""

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools_returns_text(self, mock_anthropic_class):
        """Test that response without tools returns direct text"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="This is a response")],
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(query="Hello")

        assert result == "This is a response"

    @patch("anthropic.Anthropic")
    def test_generate_response_includes_tools_in_api_call(self, mock_anthropic_class):
        """Test that tools are passed to the API when provided"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn", content=[MockContentBlock("text", text="Response")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]

        generator.generate_response(query="Test", tools=tools)

        # Verify tools were passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    @patch("anthropic.Anthropic")
    def test_generate_response_triggers_tool_execution_on_tool_use(
        self, mock_anthropic_class, mock_vector_store
    ):
        """Test that tool_use stop_reason triggers tool execution"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "machine learning"},
                    block_id="tool_123",
                )
            ],
        )

        # Second response: final answer
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Here is the answer")],
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Create tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        tools = tool_manager.get_tool_definitions()
        result = generator.generate_response(
            query="What is machine learning?", tools=tools, tool_manager=tool_manager
        )

        # Should have made two API calls
        assert mock_client.messages.create.call_count == 2
        assert result == "Here is the answer"

    @patch("anthropic.Anthropic")
    def test_tool_execution_passes_correct_parameters(
        self, mock_anthropic_class, mock_vector_store
    ):
        """Test that tool is called with correct parameters from AI"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={
                        "query": "neural networks",
                        "course_name": "Deep Learning",
                    },
                    block_id="tool_123",
                )
            ],
        )

        final_response = MockResponse(
            stop_reason="end_turn", content=[MockContentBlock("text", text="Answer")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="test-model")
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        generator.generate_response(
            query="Test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify search was called with correct params
        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name="Deep Learning", lesson_number=None
        )

    @patch("anthropic.Anthropic")
    def test_tool_result_is_passed_back_to_api(
        self, mock_anthropic_class, mock_vector_store
    ):
        """Test that tool results are included in follow-up API call"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "test"},
                    block_id="tool_123",
                )
            ],
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Final answer")],
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="test-model")
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        generator.generate_response(
            query="Test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check second API call includes tool results
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have: original query, assistant tool_use, user tool_result
        assert len(messages) == 3

        # Last message should contain tool_result
        last_message = messages[-1]
        assert last_message["role"] == "user"
        assert any(
            item.get("type") == "tool_result" for item in last_message["content"]
        )


class TestAIGeneratorSystemPrompt:
    """Tests for system prompt configuration"""

    def test_system_prompt_mentions_search_tool(self):
        """Test that system prompt includes search tool info"""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_mentions_outline_tool(self):
        """Test that system prompt includes outline tool info"""
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_includes_response_protocol(self):
        """Test that system prompt has response guidelines"""
        prompt = AIGenerator.SYSTEM_PROMPT.lower()
        assert "response" in prompt
        assert "course" in prompt


class TestAIGeneratorConversationHistory:
    """Tests for conversation history handling"""

    @patch("anthropic.Anthropic")
    def test_conversation_history_is_included_in_system(self, mock_anthropic_class):
        """Test that conversation history is added to system prompt"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn", content=[MockContentBlock("text", text="Response")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        history = "User: Previous question\nAssistant: Previous answer"

        generator.generate_response(query="New question", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        assert "Previous question" in system_content
        assert "Previous answer" in system_content

    @patch("anthropic.Anthropic")
    def test_no_history_uses_base_system_prompt(self, mock_anthropic_class):
        """Test that no history uses just the base prompt"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn", content=[MockContentBlock("text", text="Response")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Question")

        call_kwargs = mock_client.messages.create.call_args[1]
        system_content = call_kwargs["system"]

        # Should be just the base prompt without "Previous conversation" section
        assert (
            "Previous conversation" not in system_content
            or system_content == AIGenerator.SYSTEM_PROMPT
        )


class TestAIGeneratorErrorHandling:
    """Tests for error handling in AI generation"""

    @patch("anthropic.Anthropic")
    def test_handles_missing_tool_manager_gracefully(self, mock_anthropic_class):
        """Test that tool_use without tool_manager doesn't crash"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Response that would normally trigger tool use
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "test"},
                    block_id="tool_123",
                )
            ],
        )

        mock_client.messages.create.return_value = tool_use_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Call without tool_manager - should not crash but behavior depends on implementation
        # The current implementation requires tool_manager for tool execution
        result = generator.generate_response(
            query="Test",
            tools=[{"name": "test", "description": "test", "input_schema": {}}],
            tool_manager=None,
        )

        # Should return something (either the tool_use response text or handle gracefully)
        assert result is not None


class TestSequentialToolCalling:
    """Tests for sequential tool calling behavior (up to 2 rounds)"""

    @patch("anthropic.Anthropic")
    def test_two_sequential_tool_calls_makes_three_api_calls(
        self, mock_anthropic_class
    ):
        """Test that two tool rounds result in three API calls"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: first tool call
        round1_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="get_course_outline",
                    input_data={"course_title": "Course X"},
                    block_id="tool_1",
                )
            ],
        )

        # Round 2: second tool call based on first results
        round2_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search_course_content",
                    input_data={"query": "topic from lesson 4"},
                    block_id="tool_2",
                )
            ],
        )

        # Final: text response
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Found matching course")],
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Create mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Mocked tool result"

        result = generator.generate_response(
            query="Find course matching lesson 4 of Course X",
            tools=[{"name": "mock_tool"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_client.messages.create.call_count == 3
        assert result == "Found matching course"

    @patch("anthropic.Anthropic")
    def test_max_rounds_enforced_final_call_has_no_tools(self, mock_anthropic_class):
        """Test that after 2 tool rounds, final call is made without tools"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Both rounds return tool_use
        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search",
                    input_data={"query": "test"},
                    block_id="tool_x",
                )
            ],
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Final answer")],
        )

        mock_client.messages.create.side_effect = [
            tool_response,
            tool_response,
            final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator.generate_response(
            query="Test", tools=[{"name": "search"}], tool_manager=mock_tool_manager
        )

        # Verify third call has NO tools (forcing text response)
        third_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in third_call[1]

    @patch("anthropic.Anthropic")
    def test_tool_error_terminates_gracefully(self, mock_anthropic_class):
        """Test that tool execution error leads to graceful final response"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="search",
                    input_data={"query": "test"},
                    block_id="tool_err",
                )
            ],
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Sorry, encountered an issue")],
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]

        # Create tool manager that raises exception
        failing_tool_manager = MagicMock()
        failing_tool_manager.execute_tool.side_effect = Exception("Network error")

        generator = AIGenerator(api_key="test-key", model="test-model")

        result = generator.generate_response(
            query="Test", tools=[{"name": "search"}], tool_manager=failing_tool_manager
        )

        # Should still return a response (from final call without tools)
        assert result is not None
        assert mock_client.messages.create.call_count == 2

    @patch("anthropic.Anthropic")
    def test_messages_accumulate_across_rounds(self, mock_anthropic_class):
        """Test that messages array grows correctly across tool rounds"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Setup two rounds of tool use
        round1 = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use", name="tool1", input_data={}, block_id="id1"
                )
            ],
        )
        round2 = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use", name="tool2", input_data={}, block_id="id2"
                )
            ],
        )
        final = MockResponse(
            stop_reason="end_turn", content=[MockContentBlock("text", text="Done")]
        )

        mock_client.messages.create.side_effect = [round1, round2, final]

        generator = AIGenerator(api_key="test-key", model="test-model")
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator.generate_response(
            query="Q", tools=[{"name": "mock"}], tool_manager=mock_tool_manager
        )

        # Check final call has accumulated messages
        final_call = mock_client.messages.create.call_args_list[2]
        messages = final_call[1]["messages"]

        # Should have: user(original), assistant(tool1), user(result1),
        #              assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    @patch("anthropic.Anthropic")
    def test_single_tool_call_backward_compatible(
        self, mock_anthropic_class, mock_vector_store
    ):
        """Test backward compatibility - single tool call returns answer"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    name="get_course_outline",
                    input_data={"course_title": "Test"},
                    block_id="tool_single",
                )
            ],
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Here is the outline")],
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]

        generator = AIGenerator(api_key="test-key", model="test-model")
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        result = generator.generate_response(
            query="Get outline",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Only 2 API calls - no unnecessary continuation
        assert mock_client.messages.create.call_count == 2
        assert result == "Here is the outline"

    @patch("anthropic.Anthropic")
    def test_no_tool_call_returns_direct_response(self, mock_anthropic_class):
        """Test that non-tool queries work unchanged with single API call"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        direct_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Hello! How can I help?")],
        )
        mock_client.messages.create.return_value = direct_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(query="Hello")

        assert mock_client.messages.create.call_count == 1
        assert result == "Hello! How can I help?"
