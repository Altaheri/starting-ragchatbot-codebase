import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
1. **search_course_content**: Search for specific course content and educational materials
2. **get_course_outline**: Get course structure including title, link, and complete lesson list

Tool Usage:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson lists, or what topics a course covers
- **Maximum 2 sequential tool calls per query**
- Use the first tool call to gather initial information
- Use the second tool call if you need additional data based on first results
- Synthesize results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol for Course Outlines:
- When responding to outline queries, include: course title, course link, and the complete lesson list
- For each lesson, include: lesson number and lesson title
- Format the outline in a clear, readable structure

Response Protocol (General):
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2  # Maximum sequential tool call rounds per query

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with support for up to MAX_TOOL_ROUNDS sequential tool calls.

        Each round allows Claude to reason about previous results before deciding
        whether to make another tool call or provide a final answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        messages = [{"role": "user", "content": query}]
        system_content = self._build_system_content(conversation_history)

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Make API call with tools available
            response = self._make_api_call(messages, system_content, tools)

            # Exit if no tool use requested or no tool manager
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text_response(response)

            # Execute tools and collect results
            tool_results, had_error = self._execute_tools_safely(response, tool_manager)

            # Append this round to messages
            self._append_tool_round(messages, response, tool_results)

            # If error occurred, break to final call without tools
            if had_error:
                break

        # After loop (max rounds or error): final call WITHOUT tools
        return self._make_final_call(messages, system_content)

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system content with optional conversation history."""
        if conversation_history:
            return f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        return self.SYSTEM_PROMPT

    def _make_api_call(self, messages: List, system_content: str, tools: Optional[List] = None):
        """Make a single API call with optional tools."""
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _extract_text_response(self, response) -> str:
        """Extract text from response content blocks."""
        for content_block in response.content:
            if content_block.type == "text":
                return content_block.text
        return "I apologize, but I was unable to process your request. Please try again."

    def _execute_tools_safely(self, response, tool_manager) -> tuple:
        """
        Execute all tool calls from response with error handling.

        Returns:
            Tuple of (tool_results list, had_error bool)
        """
        tool_results = []
        had_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })
                    had_error = True

        return tool_results, had_error

    def _append_tool_round(self, messages: List, response, tool_results: List) -> None:
        """Append assistant tool use and user tool results to messages."""
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    def _make_final_call(self, messages: List, system_content: str) -> str:
        """Make final API call without tools to get text response."""
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text