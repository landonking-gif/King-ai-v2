"""
Master AI Brain - The central orchestrator for the autonomous empire.

This is the ONLY component that directly interfaces with the LLM (via LLMRouter).
All strategic decisions, operational planning, and agent delegation flow through this class.
It serves as the "CEO" of the business empire.
"""

import json
import asyncio
import time
from datetime import datetime
from uuid import uuid4

# Internal project imports
from src.master_ai.context import ContextManager
from src.master_ai.planner import Planner
from src.master_ai.evolution import EvolutionEngine
from src.master_ai.prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from src.master_ai.models import (
    ClassifiedIntent, IntentType, ActionType, MasterAIResponse,
    ActionResult, PlanStep
)
from src.agents.router import AgentRouter
from src.business.playbook_executor import PlaybookExecutor
from src.business.playbook_loader import PlaybookLoader
from src.business.playbook_models import TriggerType
from src.database.connection import get_db, get_db_ctx
from src.database.models import Task, EvolutionProposal
from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.structured_logging import get_logger, set_request_context
from src.utils.retry import with_retry, LLM_RETRY_CONFIG, TransientError
from src.utils.monitoring import monitor, trace_llm
from src.utils.fact_checker import check_for_hallucination
from src.utils.web_tools import get_web_tools, evaluate_simple_math
from config.settings import settings


logger = get_logger("master_ai")

class MasterAI:
    """
    The Master AI core that manages the autonomous lifecycle of businesses.
    
    Key Responsibilities:
    1. Intent Classification: Deciding if a user wants to chat, query data, or issue commands.
    2. Planning: Breaking down complex user goals into executable subtasks.
    3. Delegation: Routing subtasks to specialized agents (e.g., ResearchAgent).
    4. Autonomous Optimization: Periodic background checks to improve business performance.
    5. Evolution: Proposing source code or structural changes to itself.
    """
    
    def __init__(self):
        """Initializes all sub-components and management engines."""
        logger.info("Initializing Master AI brain")
        
        # Primary LLM interface - handles all provider routing and fallback
        self.llm_router = LLMRouter()
        
        # Specialized manager components
        self.context = ContextManager()      # Handles RAG and context window building
        self.planner = Planner(self.llm_router)  # Uses LLM router for intelligent routing
        # EvolutionEngine expects the router interface (not a raw provider client)
        self.evolution = EvolutionEngine(self.llm_router)
        self.agent_router = AgentRouter()    # Dispatches tasks to sub-agents
        self.web_tools = get_web_tools()      # Real-time web access and utilities
        
        # Performance/State tracking for conversational continuity
        self.last_subject: str | None = None
        self.last_research_query: str | None = None
        
        # State and rate limiting for autonomous features
        self.autonomous_mode = getattr(settings, 'enable_autonomous_mode', False)
        self._evolution_count_this_hour = 0
        self._hour_start = datetime.now()
        self._total_tokens_today = 0
        self._token_budget_daily = 1_000_000  # 1M tokens per day
        
        # Store last built context for fact-checking
        self._last_context: str = ""
        self._last_context_timestamp = datetime.now()
        
        # In-memory conversation history for session context
        self._conversation_history: list[dict] = []
        self._max_history_length = 20  # Keep last 20 exchanges
        
        risk_profile = getattr(settings, "risk_profile", "moderate")
        if not isinstance(risk_profile, str):
            risk_profile = "moderate"

        # Playbook system for executing business strategies
        self.playbook_loader = PlaybookLoader()
        self.playbook_executor = PlaybookExecutor()
        
        # Register agents with playbook executor
        for agent_name, agent in self.agent_router.agents.items():
            self.playbook_executor.register_agent(agent_name, agent)

        logger.info(
            "Master AI initialized",
            autonomous_mode=self.autonomous_mode,
            risk_profile=risk_profile
        )
    
    async def process_input(self, user_input: str, request_id: str = None) -> MasterAIResponse:
        """
        Primary entry point for any user interaction (API or CLI).
        
        Logic:
        1. Build the current global context (financials, active tasks, history).
        2. Use the LLM to classify the user's intent.
        3. Dispatch to the appropriate handler (conversation, command, or query).
        
        Args:
            user_input: The user's message
            request_id: Optional request ID for tracing
            
        Returns:
            Structured MasterAIResponse
        """
        request_id = request_id or str(uuid4())
        set_request_context(request_id)
        start_time = time.time()
        
        logger.info("Processing user input", input_length=len(user_input))
        monitor.increment("master_ai.requests")
        
        try:
            # Check for simple queries we can handle directly without LLM
            direct_response = await self._try_direct_response(user_input)
            if direct_response:
                self._log_conversation(user_input, direct_response)
                return MasterAIResponse(
                    type="conversation",
                    response=direct_response,
                    metadata={"handled_directly": True}
                )
            
            # 1. Gather all relevant data for the LLM to make informed decisions
            context = await self.context.build_context(query=user_input)
            
            # Store context for fact-checking
            self._last_context = context
            self._last_context_timestamp = datetime.now()
            
            # 2. Classify the user's intent with structured output
            intent = await self._classify_intent(user_input, context)
            logger.info(
                "Intent classified",
                intent_type=intent.type.value,
                action=intent.action.value if intent.action else None,
                confidence=intent.confidence
            )
            
            # 3. Route to appropriate handler based on intent
            if intent.type == IntentType.CONVERSATION:
                response = await self._handle_conversation(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
                
            elif intent.type == IntentType.COMMAND:
                result = await self._handle_command(user_input, intent, context)
                
            elif intent.type == IntentType.QUERY:
                response = await self._handle_query(user_input, context)
                result = MasterAIResponse(
                    type="conversation",
                    response=response,
                    metadata={"intent_confidence": intent.confidence}
                )
            else:
                result = MasterAIResponse(
                    type="error",
                    response="Unknown intent type"
                )
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("master_ai.request_duration", duration_ms)
            monitor.increment("master_ai.requests_success")
            
            # Log conversation for context
            self._log_conversation(user_input, result.response)
            
            logger.info(
                "Request completed",
                duration_ms=duration_ms,
                response_type=result.type
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("master_ai.requests_error")
            logger.error("Request failed", exc_info=True, error=str(e))
            
            return MasterAIResponse(
                type="error",
                response=f"I encountered an error processing your request: {str(e)}",
                metadata={"error": str(e), "duration_ms": duration_ms}
            )

    @with_retry(LLM_RETRY_CONFIG)
    async def _classify_intent(self, user_input: str, context: str) -> ClassifiedIntent:
        """
        Uses the LLM as a router to interpret natural language into a structured intent.
        
        Returns a validated ClassifiedIntent object.
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_input=user_input,
            context=context[:2000]  # Limit context for classification
        )
        
        # Use high-accuracy context for classification as it's critical for routing
        task_context = TaskContext(
            task_type="classification",
            risk_level="low",
            requires_accuracy=True,
            token_estimate=500,
            priority="normal"
        )
        
        response = await self._call_llm(prompt, task_context=task_context)
        
        try:
            # Parse and validate the response
            parsed = self._parse_json_response(response)
            
            return ClassifiedIntent(
                type=IntentType(parsed.get("type", "conversation")),
                action=ActionType(parsed.get("action")) if parsed.get("action") else None,
                parameters=parsed.get("parameters", {}),
                confidence=parsed.get("confidence", 0.5),
                requires_planning=parsed.get("requires_planning", False),
                reasoning=parsed.get("reasoning")
            )
        except Exception as e:
            # Smart fallback: detect action-oriented requests
            user_lower = user_input.lower()
            
            # Keywords that suggest the user wants ACTION, not conversation
            action_keywords = [
                "create", "build", "start", "make", "launch", "setup", "set up",
                "research", "find", "analyze", "investigate", "look up", "search",
                "generate", "produce", "write", "develop", "implement",
                "stop", "shut down", "cancel", "pause",
                "optimize", "improve", "fix", "update",
                "do this", "do it", "just do", "execute", "run"
            ]
            
            is_action_request = any(kw in user_lower for kw in action_keywords)
            
            if is_action_request:
                logger.info(
                    "Intent parse failed, detected action keywords - routing to COMMAND",
                    keywords_found=[kw for kw in action_keywords if kw in user_lower]
                )
                
                # Determine the best action based on keywords
                if any(kw in user_lower for kw in ["research", "find", "analyze", "investigate", "look up", "search"]):
                    action = ActionType.RESEARCH_MARKET
                elif any(kw in user_lower for kw in ["create", "build", "start", "make", "launch", "setup"]):
                    action = ActionType.START_BUSINESS
                elif any(kw in user_lower for kw in ["optimize", "improve"]):
                    action = ActionType.OPTIMIZE_BUSINESS
                elif any(kw in user_lower for kw in ["generate", "write"]):
                    action = ActionType.CREATE_CONTENT
                else:
                    action = ActionType.UNKNOWN
                
                return ClassifiedIntent(
                    type=IntentType.COMMAND,
                    action=action,
                    confidence=0.6,
                    requires_planning=True,
                    reasoning=f"Detected action intent from keywords, bypassing failed parse."
                )

            logger.warning(
                "Failed to parse intent, defaulting to conversation",
                error=str(e),
                response=response[:200]
            )
            return ClassifiedIntent(type=IntentType.CONVERSATION, confidence=0.3)
    
    async def _try_direct_response(self, user_input: str) -> str | None:
        """
        Try to handle queries directly without LLM for speed and accuracy.
        
        Handles:
        - Date/time questions
        - Simple math
        - Greetings and identity
        - Crypto prices
        - Weather
        - Wikipedia lookups
        - Unit conversions
        - Agent capabilities
        
        Returns:
            Response string if handled directly, None otherwise
        """
        from src.utils.web_tools import (
            get_crypto_price, get_weather, get_wikipedia_summary,
            convert_units, parse_unit_conversion_query
        )
        import re
        
        user_lower = user_input.lower().strip()
        
        # === DATE/TIME QUERIES ===
        datetime_keywords = ["what time", "what date", "what day", "current time", 
                            "current date", "date and time", "time and date", "what's the time",
                            "what is the time", "what is the date", "today's date", "what year"]
        
        if any(kw in user_lower for kw in datetime_keywords):
            dt_info = self.web_tools.get_current_datetime()
            return f"The current date is {dt_info['day_of_week']}, {dt_info['date']} and the time is {dt_info['time']}."
        
        # === SIMPLE MATH ===
        math_result = evaluate_simple_math(user_input)
        if math_result is not None:
            return f"The answer is {math_result}."
        
        # === UNIT CONVERSIONS ===
        conversion = parse_unit_conversion_query(user_input)
        if conversion:
            result = convert_units(conversion["value"], conversion["from_unit"], conversion["to_unit"])
            if result is not None:
                if result == int(result):
                    result = int(result)
                return f"{conversion['value']} {conversion['from_unit']} = {result:.2f if isinstance(result, float) else result} {conversion['to_unit']}"
        
        # === CRYPTO PRICES (General) ===
        # bitcoin, eth, solana price, value of bitcoin
        crypto_match = re.search(r"(?:price|value) of (bitcoin|ethereum|litecoin|cardano|solana|polkadot|dogecoin|btc|eth|sol|doge|crypto)", user_lower)
        if crypto_match or any(kw in user_lower for kw in ["bitcoin", "ethereum", "btc price", "eth price"]):
            symbol = "bitcoin" if "bitcoin" in user_lower or "btc" in user_lower else \
                     "ethereum" if "ethereum" in user_lower or "eth" in user_lower else \
                     "solana" if "solana" in user_lower or "sol" in user_lower else \
                     "dogecoin" if "dogecoin" in user_lower or "doge" in user_lower else None
            
            if crypto_match and not symbol:
                symbol = crypto_match.group(1).strip()
                shortcuts = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "doge": "dogecoin"}
                symbol = shortcuts.get(symbol, symbol)

            if symbol and symbol != "crypto":
                try:
                    crypto_data = await get_crypto_price(symbol)
                    if crypto_data:
                        direction = "▲" if crypto_data["change_24h"] >= 0 else "▼"
                        return f"{crypto_data['symbol']}: ${crypto_data['price_usd']:,.2f} ({direction} {abs(crypto_data['change_24h']):.2f}% 24h)"
                except Exception:
                    pass

        # === STOCK PRICES (General) ===
        # "price of AAPL", "price of tesla", "stock price of google", "what about walmart"
        # Match "what about X" ONLY if the last subject was a stock/crypto
        stock_patterns = [
            r"(?:price|stock|value|shares) (?:of|for|on) ([a-zA-Z0-9\.\^' ]{2,20})",
            r"([a-zA-Z0-9\.\^' ]{2,20})'s (?:stock |)(?:price|shares|value|stock)",
            r"what about ([a-zA-Z0-9\.\^' ]{2,20})"
        ]
        
        target_symbol = None
        for pattern in stock_patterns:
            match = re.search(pattern, user_lower)
            if match:
                candidate = match.group(1).strip().rstrip("?.")
                # If "what about X", we check if we have a previous subject or if it looks like a name
                if "what about" in pattern and not self.last_subject and len(candidate) > 6:
                    continue
                target_symbol = candidate
                break

        if target_symbol:
            # Filter out common non-stock words
            if target_symbol not in ["bitcoin", "ethereum", "crypto", "it", "this", "that", "weather", "today", "tomorrow"]:
                try:
                    target_symbol = target_symbol.rstrip("'s").rstrip("'")
                    # Expanded map for "Limitless" feel
                    common_map = {
                        "tesla": "TSLA", "apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT", 
                        "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
                        "walmart": "WMT", "wallmart": "WMT", "target": "TGT", "disney": "DIS",
                        "netflix": "NFLX", "spotify": "SPOT", "uber": "UBER", "lyft": "LYFT",
                        "airbnb": "ABNB", "twitter": "X", "openai": "MSFT", "spacex": "TSLA",
                        "amd": "AMD", "intel": "INTC", "tsmc": "TSM", "arm": "ARM"
                    }
                    ticker = common_map.get(target_symbol, target_symbol)
                    stock_data = await self.web_tools.get_stock_price(ticker)
                    if stock_data:
                        self.last_subject = ticker
                        direction = "▲" if stock_data.change >= 0 else "▼"
                        return f"{stock_data.symbol}: ${stock_data.price:,.2f} ({direction} {stock_data.change_percent:.2f}%)"
                except Exception:
                    pass

        # === RESEARCH / SEARCH COMMANDS ===
        # "research the market", "analyze the data", "analyze"
        if user_lower.startswith(("research ", "search for ", "find info on ", "analyze ")) or user_lower == "analyze":
            query = user_lower.replace("research ", "").replace("search for ", "").replace("find info on ", "").strip()
            
            # Carry over last query for "analyze"
            if user_lower == "analyze" or query == "analyze":
                if self.last_research_query:
                    query = self.last_research_query
                else:
                    return "What would you like me to analyze? (e.g., 'research the AI market' then 'analyze')"
            
            if len(query) > 2:
                self.last_research_query = query
                # Delegate directly to research agent via web search tool
                try:
                    results = await self.web_tools.search_formatted(query, max_results=3)
                    return f"Here is what I found for '{query}':\n\n{results}\n\n(type 'analyze' for deep synthesis of this topic)"
                except Exception:
                    pass
        
        # === WEATHER ===
        weather_patterns = ["weather in", "weather for", "what's the weather", "how's the weather", "temperature in"]
        for pattern in weather_patterns:
            if pattern in user_lower:
                try:
                    # Extract city name
                    city = "New York"  # default
                    if "weather in " in user_lower:
                        city = user_input.split("weather in ")[-1].strip().rstrip("?.")
                    elif "weather for " in user_lower:
                        city = user_input.split("weather for ")[-1].strip().rstrip("?.")
                    elif "temperature in " in user_lower:
                        city = user_input.split("temperature in ")[-1].strip().rstrip("?.")
                    
                    weather_data = await get_weather(city)
                    if weather_data:
                        return f"Weather in {weather_data['city']}: {weather_data['condition']}, {weather_data['temperature_f']}°F ({weather_data['temperature_c']}°C), Humidity: {weather_data['humidity']}%, Wind: {weather_data['wind_mph']} mph"
                except Exception:
                    pass
                break
        
        # === WIKIPEDIA LOOKUPS ===
        wiki_patterns = ["what is a ", "what is an ", "what is the ", "define ", "explain ", "tell me about "]
        for pattern in wiki_patterns:
            if user_lower.startswith(pattern):
                topic = user_input[len(pattern):].strip().rstrip("?.!")
                if len(topic) > 2:
                    try:
                        summary = await get_wikipedia_summary(topic)
                        if summary:
                            # Truncate if too long
                            if len(summary) > 500:
                                summary = summary[:500] + "..."
                            return summary
                    except Exception:
                        pass
                break
        
        # === GREETINGS ===
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_lower in greetings or user_lower.rstrip("!?.") in greetings:
            dt_info = self.web_tools.get_current_datetime()
            hour = int(dt_info['time'].split(":")[0])
            is_pm = "PM" in dt_info['time']
            if is_pm and hour != 12:
                hour += 12
            if not is_pm and hour == 12:
                hour = 0
            time_of_day = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
            return f"Good {time_of_day}! I'm King AI. How can I help you today?"
        
        # === NEWS HEADLINES ===
        if user_lower.startswith(("news on ", "latest news", "top stories")):
            topic = user_lower.replace("news on ", "").replace("latest news", "business").replace("top stories", "general").strip()
            try:
                news_text = await self.web_tools.get_news_formatted(topic)
                return f"Here are the latest headlines for '{topic}':\n\n{news_text}"
            except Exception:
                pass

        # === IDENTITY ===
        if "who are you" in user_lower or "what are you" in user_lower:
            return "I'm King AI, an autonomous business management AI. I can help you with business strategy, market research, financial analysis, content creation, web search, stock prices, crypto prices, weather, and much more. What would you like to know or do?"
        
        # === CAPABILITIES ===
        if "what can you do" in user_lower or "help" == user_lower or "capabilities" in user_lower:
            return """I'm King AI and I can help you with:

📊 **Real-time Data**: Stock prices, crypto prices, market trends, weather
🔍 **Research**: Web search, Wikipedia lookups, market research
📝 **Business**: Strategy planning, financial analysis, content creation
🧮 **Calculations**: Math, unit conversions, currency exchange
🤖 **Agent Tasks**: Research, analytics, content, commerce, finance, legal

Just ask me anything! For example:
- "What's the weather in Chicago?"
- "What is Bitcoin's price?"
- "Convert 100 miles to km"
- "What is machine learning?"
- "Research the AI industry"
"""
        
        # === LIST AGENTS ===
        if "list agents" in user_lower or "what agents" in user_lower or "available agents" in user_lower:
            agents = self.agent_router.list_agents()
            response = "**Available Agents:**\n\n"
            for agent in agents:
                response += f"• **{agent['name']}** ({agent['risk_level']} risk)\n"
            return response
        
        # Not a simple query - let the normal flow handle it
        return None

    async def _delegate_to_agent(self, agent_name: str, task_description: str, parameters: dict = None) -> dict:
        """
        Delegate a task to a specialized agent.
        
        Args:
            agent_name: Name of the agent (research, analytics, content, etc.)
            task_description: Description of what to do
            parameters: Additional task parameters
            
        Returns:
            Agent result dict
        """
        task = {
            "agent": agent_name,
            "description": task_description,
            "action": task_description,
            **(parameters or {})
        }
        
        try:
            result = await self.agent_router.execute(task)
            logger.info(
                "Agent task completed",
                agent=agent_name,
                success=result.get("success", False)
            )
            return result
        except Exception as e:
            logger.error(f"Agent delegation failed: {e}", agent=agent_name)
            return {"success": False, "error": str(e)}
    
    async def execute_research(self, query: str) -> str:
        """
        Execute a research query using the research agent.
        
        Args:
            query: Research topic or question
            
        Returns:
            Research results formatted as string
        """
        result = await self._delegate_to_agent("research", query, {
            "query": query,
            "research_type": "general"
        })
        
        if result.get("success"):
            output = result.get("output", {})
            if isinstance(output, dict):
                summary = output.get("summary", str(output))
                findings = output.get("key_findings", [])
                if findings:
                    summary += "\n\nKey Findings:\n" + "\n".join(f"• {f}" for f in findings)
                return summary
            return str(output)
        return f"Research failed: {result.get('error', 'Unknown error')}"
    
    async def execute_analytics(self, query: str) -> str:
        """
        Execute an analytics query using the analytics agent.
        
        Args:
            query: Analytics query
            
        Returns:
            Analytics results formatted as string
        """
        result = await self._delegate_to_agent("analytics", query, {
            "query": query,
            "analysis_type": "general"
        })
        
        if result.get("success"):
            return str(result.get("output", "Analysis complete"))
        return f"Analytics failed: {result.get('error', 'Unknown error')}"

    
    def _log_conversation(self, user_input: str, response: str):
        """Log conversation exchange to in-memory history."""
        self._conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        self._conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim to max length
        if len(self._conversation_history) > self._max_history_length * 2:
            self._conversation_history = self._conversation_history[-self._max_history_length * 2:]
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for inclusion in prompts."""
        if not self._conversation_history:
            return "No previous conversation."
        
        lines = []
        for msg in self._conversation_history[-10:]:  # Last 5 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]  # Truncate long messages
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    async def _handle_conversation(self, user_input: str, context: str) -> str:
        """Handles conversation with real-time context integration."""
        
        # Get conversation history
        history = self._format_conversation_history()
        
        # Build real-time context section
        realtime_context = self.web_tools.get_datetime_context()
        
        # Check if query needs web data
        needs = self.web_tools.detect_realtime_query(user_input)
        
        if needs["needs_web_search"] or needs["needs_market_trends"]:
            try:
                web_results = await self.web_tools.search_formatted(user_input, max_results=3)
                realtime_context += f"\n\n{web_results}"
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        if needs["needs_stock_data"] or needs["needs_market_trends"]:
            try:
                market_summary = await self.web_tools.get_market_summary()
                realtime_context += f"\n\n{market_summary}"
            except Exception as e:
                logger.warning(f"Market data fetch failed: {e}")
        
        prompt = f"""{SYSTEM_PROMPT}

=== REAL-TIME INFORMATION ===
{realtime_context}

=== BUSINESS CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{history}

=== CURRENT USER MESSAGE ===
{user_input}

=== INSTRUCTIONS ===
Respond helpfully to the user's message. You have access to:
1. Current date/time from REAL-TIME INFORMATION
2. Business data from BUSINESS CONTEXT
3. Web search results (if provided above)
4. Market data (if provided above)

Guidelines:
- Use the real-time information when relevant
- For business questions, use the BUSINESS CONTEXT
- If information is not available in any section, say so honestly
- Do NOT make up facts, numbers, or information
- Be conversational and helpful
"""
        task_context = TaskContext(
            task_type="conversation",
            risk_level="low",
            requires_accuracy=True,
            token_estimate=1000,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)
    
    async def _handle_command(
        self,
        user_input: str,
        intent: ClassifiedIntent,
        context: str
    ) -> MasterAIResponse:
        """
        Orchestrates complex actions:
        1. Breaks goal into steps via the Planner.
        2. Checks for approval requirements based on Risk Profile.
        3. Executes immediate tasks or queues for user review.
        """
        logger.info(
            "Handling command",
            action=intent.action.value if intent.action else "unknown",
            parameters=intent.parameters
        )
        
        # Create a detailed multi-step plan
        plan = await self.planner.create_plan(
            goal=user_input,
            action=intent.action.value if intent.action else None,
            parameters=intent.parameters,
            context=context
        )
        
        actions_taken = []
        pending_approvals = []
        
        # Iterate through planned steps
        for step in plan.get("steps", []):
            # Convert dict to PlanStep if needed
            if isinstance(step, dict):
                step_model = PlanStep(
                    name=step.get("name", "unknown"),
                    description=step.get("description", ""),
                    agent=step.get("agent", "general"),
                    requires_approval=step.get("requires_approval", False),
                    dependencies=step.get("dependencies", []),
                    estimated_duration=step.get("estimated_duration", "unknown"),
                    input_data=step.get("input", {}),
                    risk_level=step.get("risk_level", "low")
                )
            else:
                step_model = step
            
            if step_model.requires_approval:
                # If high risk, save to DB for user review
                task_info = await self._create_approval_task(step_model)
                pending_approvals.append(task_info)
                logger.info("Task queued for approval", task_name=step_model.name)
            else:
                # If low risk, execute immediately via router
                start = time.time()
                result = await self.agent_router.execute(step if isinstance(step, dict) else step.dict())
                duration_ms = (time.time() - start) * 1000
                
                action_result = ActionResult(
                    step_name=step_model.name,
                    agent=step_model.agent,
                    success=result.get("success", False),
                    output=result.get("output"),
                    error=result.get("error"),
                    duration_ms=duration_ms
                )
                actions_taken.append(action_result)
                
                logger.agent_execution(
                    agent=step_model.agent,
                    task=step_model.name,
                    duration_ms=duration_ms,
                    success=result.get("success", False)
                )
        
        # Generate a final confirmation or status report for the user
        response = await self._generate_action_summary(
            user_input,
            [a.dict() for a in actions_taken],
            pending_approvals,
            context
        )
        
        return MasterAIResponse(
            type="action",
            response=response,
            actions_taken=actions_taken,
            pending_approvals=pending_approvals,
            metadata={
                "plan_steps": len(plan.get("steps", [])),
                "executed": len(actions_taken),
                "pending": len(pending_approvals)
            }
        )
    
    async def _generate_action_summary(
        self,
        user_input: str,
        actions: list,
        approvals: list,
        context: str
    ) -> str:
        """Uses LLM to summarize what was just done and what is pending."""
        prompt = f"""Summarize the actions taken and items pending approval in response to: "{user_input}"
        
Actions Completed: {json.dumps(actions, default=str)}
Pending Review: {json.dumps(approvals, default=str)}

Provide a concise confirmation message to the user. Be specific about what was done and what needs approval.
"""
        task_context = TaskContext(
            task_type="summary",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=500,
            priority="normal"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    async def _handle_query(self, user_input: str, context: str) -> str:
        """Answers data queries ONLY using context data - no fabrication."""
        
        # Get conversation history
        history = self._format_conversation_history()
        
        prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT (This is ALL the information you have access to) ===
{context}

=== CONVERSATION HISTORY ===
{history}

=== USER QUERY ===
{user_input}

=== STRICT INSTRUCTIONS ===
1. ONLY answer using information explicitly stated in the CONTEXT section above
2. If the information is NOT in the context, respond: "I don't have that information in my current context."
3. Do NOT make up statistics, metrics, numbers, or any other data
4. Do NOT claim to access external files, databases, or systems
5. Do NOT provide information about current events, news, or real-time data
6. Be honest and direct about what you do and don't know
"""
        task_context = TaskContext(
            task_type="query",
            risk_level="high",  # Queries need maximum accuracy
            requires_accuracy=True,
            token_estimate=1000,
            priority="high"
        )
        
        return await self._call_llm(prompt, task_context=task_context)

    # =========================================================================
    # PLAYBOOK EXECUTION SYSTEM
    # =========================================================================
    
    async def execute_playbook(
        self,
        playbook_id: str,
        business_id: str,
        context: dict = None,
        trigger: TriggerType = TriggerType.SCHEDULED,
    ) -> dict:
        """
        Execute a playbook for a business unit.
        
        This is the primary entry point for running business playbooks.
        The MasterAI coordinates the execution, delegating tasks to agents.
        
        Args:
            playbook_id: ID of the playbook to execute (e.g., 'dropshipping', 'saas')
            business_id: ID of the business unit to execute for
            context: Optional context data for the playbook
            trigger: What triggered this execution
            
        Returns:
            Execution result with status, completed tasks, and any errors
        """
        logger.info(
            "Executing playbook",
            playbook_id=playbook_id,
            business_id=business_id,
            trigger=trigger.value
        )
        
        try:
            # Load the playbook definition
            playbook = self.playbook_loader.load_playbook(playbook_id)
            if not playbook:
                return {
                    "success": False,
                    "error": f"Playbook '{playbook_id}' not found",
                    "playbook_id": playbook_id,
                    "business_id": business_id
                }
            
            # Build execution context
            exec_context = context or {}
            exec_context["business_id"] = business_id
            exec_context["triggered_at"] = datetime.now().isoformat()
            
            # Execute the playbook
            run = await self.playbook_executor.execute(
                playbook=playbook,
                business_id=business_id,
                context=exec_context,
                trigger=trigger
            )
            
            # Log completion
            logger.info(
                "Playbook execution complete",
                run_id=run.id,
                status=run.status.value,
                completed_tasks=len(run.completed_tasks),
                failed_tasks=len(run.failed_tasks)
            )
            
            return {
                "success": run.status.value == "completed",
                "run_id": run.id,
                "playbook_id": playbook_id,
                "business_id": business_id,
                "status": run.status.value,
                "completed_tasks": run.completed_tasks,
                "failed_tasks": run.failed_tasks,
                "error": run.error,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            }
            
        except Exception as e:
            logger.error("Playbook execution failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "playbook_id": playbook_id,
                "business_id": business_id
            }
    
    async def execute_playbook_stage(
        self,
        playbook_id: str,
        business_id: str,
        stage: str,
        context: dict = None
    ) -> dict:
        """
        Execute only a specific stage of a playbook.
        
        Useful for resuming playbooks or running specific phases.
        
        Args:
            playbook_id: ID of the playbook
            business_id: ID of the business unit
            stage: Stage name to execute (e.g., 'discovery', 'validation')
            context: Optional context data
            
        Returns:
            Execution result for the stage
        """
        logger.info(
            "Executing playbook stage",
            playbook_id=playbook_id,
            business_id=business_id,
            stage=stage
        )
        
        try:
            playbook = self.playbook_loader.load_playbook(playbook_id)
            if not playbook:
                return {
                    "success": False,
                    "error": f"Playbook '{playbook_id}' not found"
                }
            
            # Filter tasks to only those in the requested stage
            stage_tasks = [t for t in playbook.tasks if t.metadata.get("stage") == stage]
            
            if not stage_tasks:
                return {
                    "success": False,
                    "error": f"No tasks found for stage '{stage}' in playbook '{playbook_id}'"
                }
            
            # Create a modified playbook with only stage tasks
            from src.business.playbook_models import PlaybookDefinition
            stage_playbook = PlaybookDefinition(
                id=f"{playbook.id}_{stage}",
                name=f"{playbook.name} - {stage.title()}",
                description=f"Stage execution: {stage}",
                version=playbook.version,
                tasks=stage_tasks,
                metadata={**playbook.metadata, "original_playbook": playbook.id, "stage": stage}
            )
            
            # Execute the stage playbook
            run = await self.playbook_executor.execute(
                playbook=stage_playbook,
                business_id=business_id,
                context=context or {},
                trigger=TriggerType.SCHEDULED
            )
            
            return {
                "success": run.status.value == "completed",
                "run_id": run.id,
                "stage": stage,
                "completed_tasks": run.completed_tasks,
                "failed_tasks": run.failed_tasks,
                "error": run.error
            }
            
        except Exception as e:
            logger.error("Stage execution failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "stage": stage
            }
    
    def list_available_playbooks(self) -> list[dict]:
        """List all available playbooks with their metadata."""
        return self.playbook_loader.list_playbooks()

    @trace_llm
    async def _call_llm(
        self,
        prompt: str,
        system: str = None,
        task_context: TaskContext = None
    ) -> str:
        """
        Central LLM calling method with routing, retry, and metrics.
        All LLM calls should go through this method.
        """
        start_time = time.time()
        
        # Check token budget
        if self._total_tokens_today >= self._token_budget_daily:
            logger.warning("Daily token budget exhausted")
            raise TransientError("Daily token budget exhausted. Try again tomorrow.")
        
        try:
            response = await self.llm_router.complete(
                prompt=prompt,
                system=system,
                context=task_context
            )
            
            # Estimate tokens (rough: 4 chars per token)
            estimated_tokens = (len(prompt) + len(response)) // 4
            self._total_tokens_today += estimated_tokens
            
            duration_ms = (time.time() - start_time) * 1000
            monitor.timing("llm.call_duration", duration_ms)
            monitor.increment("llm.calls_success")
            
            # Fact-check the response for hallucinations
            if task_context and task_context.requires_accuracy:
                fact_check = check_for_hallucination(
                    response=response,
                    context=self._last_context  # Use the actual context, not the prompt
                )
                
                if not fact_check.is_valid:
                    logger.warning(
                        "Potential hallucination detected in LLM response",
                        confidence=fact_check.confidence,
                        issues=fact_check.issues,
                        warnings=fact_check.warnings,
                        task_type=task_context.task_type
                    )
                    monitor.increment("llm.hallucination_detected")
                    
                    # For critical tasks, reject hallucinated responses
                    if task_context.risk_level in ["high", "critical"]:
                        if fact_check.confidence < 0.5:
                            raise ValueError(
                                f"LLM response failed fact-check with confidence {fact_check.confidence}. "
                                f"Issues: {', '.join(fact_check.issues)}"
                            )
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(response) // 4,
                latency_ms=duration_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            monitor.increment("llm.calls_error")
            
            logger.llm_call(
                provider="router",
                model=settings.ollama_model,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=0,
                latency_ms=duration_ms,
                success=False,
                error=str(e)
            )
            
            raise
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Clean up markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        # Try to find JSON block if raw string still fails
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Look for the first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(response[start:end+1])
                except json.JSONDecodeError:
                    pass
            raise
    
    async def run_autonomous_loop(self):
        """
        Background optimization loop.
        Runs every 6 hours to analyze business units and propose evolutions.
        """
        logger.info("Starting autonomous loop")
        
        while self.autonomous_mode:
            try:
                context = await self.context.build_context()
                
                # Step 1: Self-improvement analysis
                await self._consider_evolution(context)
                
                # Step 2: Business unit health check
                await self._check_business_health(context)
                
                logger.info("Autonomous loop iteration complete")
                
            except Exception as e:
                logger.error("Autonomous loop error", exc_info=True)
            
            # Wait for the next optimization cycle (use configurable interval)
            interval_hours = getattr(settings, 'kpi_review_interval_hours', 6)
            await asyncio.sleep(interval_hours * 60 * 60)
    
    async def _check_business_health(self, context: str):
        """
        Analyze business unit performance and suggest optimizations.
        Checks health metrics across all active businesses and triggers
        automated responses for underperforming units.
        """
        from src.database.connection import get_db_ctx
        from src.database.models import BusinessUnit
        from src.business.lifecycle import LifecycleEngine
        from datetime import datetime, timedelta
        
        try:
            async with get_db_ctx() as session:
                # Get all active businesses
                from sqlalchemy import select
                from src.database.models import BusinessStatus
                
                result = await session.execute(
                    select(BusinessUnit).where(
                        BusinessUnit.status.in_([
                            BusinessStatus.LAUNCH.value,
                            BusinessStatus.OPTIMIZATION.value,
                            BusinessStatus.SCALING.value
                        ])
                    )
                )
                businesses = result.scalars().all()
                
                health_issues = []
                
                for business in businesses:
                    # Check revenue trends
                    if hasattr(business, 'monthly_revenue') and business.monthly_revenue:
                        if business.monthly_revenue < 100:  # Very low revenue
                            health_issues.append({
                                "business_id": business.id,
                                "issue": "low_revenue",
                                "severity": "high",
                                "suggested_action": "review_marketing_strategy"
                            })
                    
                    # Check for stale businesses (no activity in 7 days)
                    if business.updated_at:
                        days_inactive = (datetime.utcnow() - business.updated_at).days
                        if days_inactive > 7:
                            health_issues.append({
                                "business_id": business.id,
                                "issue": "stale_business",
                                "severity": "medium",
                                "days_inactive": days_inactive,
                                "suggested_action": "review_and_reactivate"
                            })
                    
                    # Check profitability
                    if hasattr(business, 'profit_margin') and business.profit_margin:
                        if business.profit_margin < 10:
                            health_issues.append({
                                "business_id": business.id,
                                "issue": "low_margin",
                                "severity": "high",
                                "suggested_action": "optimize_pricing_or_costs"
                            })
                
                # Log and take action on critical issues
                if health_issues:
                    logger.warning(
                        "Business health issues detected",
                        issue_count=len(health_issues),
                        high_severity=len([i for i in health_issues if i["severity"] == "high"])
                    )
                    
                    # Queue tasks for critical issues
                    for issue in health_issues:
                        if issue["severity"] == "high":
                            await self._queue_health_remediation(issue)
                else:
                    logger.info("All businesses healthy", business_count=len(businesses))
                    
        except Exception as e:
            logger.error("Business health check failed", error=str(e))
    
    async def _queue_health_remediation(self, issue: dict):
        """Queue a remediation task for a business health issue."""
        logger.info(
            "Queuing health remediation",
            business_id=issue["business_id"],
            issue=issue["issue"],
            action=issue["suggested_action"]
        )
        # In a full implementation, this would create a task for the router
    
    async def _consider_evolution(self, context: str):
        """
        Calls the EvolutionEngine to see if any system improvements are needed.
        Rate limited to prevent runaway self-modifications.
        """
        if not settings.enable_self_modification:
            return
        
        # Periodic reset of the hourly evolution counter
        now = datetime.now()
        if (now - self._hour_start).seconds >= 3600:
            self._evolution_count_this_hour = 0
            self._hour_start = now
        
        max_evolutions = getattr(settings, "max_evolutions_per_hour", 5)
        if not isinstance(max_evolutions, int):
            max_evolutions = 5

        # Enforce rate limit from settings
        if self._evolution_count_this_hour >= max_evolutions:
            logger.debug("Evolution rate limit reached")
            return
        
        # Build context for evolution proposal
        evolution_context = {
            "business_state": context,
            "recent_actions": [],
            "performance_metrics": {},
            "risk_profile": getattr(settings, 'risk_profile', 'moderate')
        }
        
        # Ask LLM for a proposal
        proposal = await self.evolution.propose_improvement(
            context=str(evolution_context),
            goal="Improve system performance and efficiency"
        )
        
        # Handle both EvolutionProposal object and legacy dict format
        from src.master_ai.evolution_models import EvolutionProposal, ProposalStatus
        
        if proposal:
            self._evolution_count_this_hour += 1
            
            # Check if it's the new EvolutionProposal type or legacy dict
            if isinstance(proposal, EvolutionProposal):
                is_beneficial = proposal.status != ProposalStatus.REJECTED
                confidence = proposal.confidence_score.overall if proposal.confidence_score else 0.0
                proposal_data = {
                    "id": proposal.id,
                    "type": proposal.proposal_type.value,
                    "description": proposal.description,
                    "changes": {c.file_path: c.new_content for c in proposal.changes if c.new_content},
                    "confidence": confidence,
                    "is_beneficial": is_beneficial
                }
            else:
                # Legacy dict format
                is_beneficial = proposal.get("is_beneficial", False)
                confidence = proposal.get("confidence", 0)
                proposal_data = proposal
            
            if not is_beneficial:
                logger.info("Evolution proposal not beneficial, skipping")
                return
            
            # Validate confidence threshold
            evolution_threshold = getattr(settings, "evolution_confidence_threshold", 0.8)
            if not isinstance(evolution_threshold, (int, float)):
                evolution_threshold = 0.8
            if confidence < evolution_threshold:
                logger.info(
                    "Evolution proposal rejected due to low confidence",
                    confidence=confidence,
                    threshold=evolution_threshold
                )
                return
            
            # Process in sandbox
            await self._process_evolution_proposal(proposal_data)
    
    async def _process_evolution_proposal(self, proposal: dict):
        """Process a validated evolution proposal through sandbox testing."""
        from src.utils.sandbox import Sandbox
        
        changes = proposal.get("changes", {})
        if not changes:
            return
        
        logger.info(
            "Processing evolution proposal",
            type=proposal.get("type"),
            files=list(changes.keys())
        )
        
        # Create sandbox and test
        sandbox = Sandbox()
        sandbox.create_sandbox(list(changes.keys()))
        
        try:
            all_patches_applied = True
            for file_path, new_code in changes.items():
                if not sandbox.apply_patch(file_path, new_code):
                    all_patches_applied = False
                    break
            
            if all_patches_applied:
                test_result = sandbox.run_tests()
                if test_result["success"]:
                    proposal["confidence"] = min(0.95, proposal.get("confidence", 0) + 0.1)
                    await self._save_evolution_proposal(proposal)
                    logger.info("Evolution proposal saved for approval")
                else:
                    logger.warning(
                        "Evolution proposal failed sandbox tests",
                        output=test_result.get("output", "")[:500]
                    )
        finally:
            sandbox.cleanup()
    
    async def _create_approval_task(self, step: PlanStep) -> dict:
        """Persists a high-risk task that requires human intervention."""
        async with get_db_ctx() as db:
            task = Task(
                id=str(uuid4()),
                name=step.name,
                description=step.description,
                type=step.agent,
                status="pending_approval",
                agent=step.agent,
                input_data=step.input_data,
                requires_approval=True
            )
            db.add(task)
            await db.commit()
            
            logger.info("Created approval task", task_id=task.id, name=task.name)
            
            return {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "agent": task.agent
            }
    
    async def _save_evolution_proposal(self, proposal: dict):
        """Saves a self-modification proposal for user approval."""
        async with get_db_ctx() as db:
            prop = EvolutionProposal(
                id=str(uuid4()),
                type=proposal["type"],
                description=proposal["description"],
                rationale=proposal["rationale"],
                proposed_changes=proposal["changes"],
                expected_impact=proposal.get("expected_impact"),
                confidence_score=proposal.get("confidence", 0.0)
            )
            db.add(prop)
            await db.commit()
            
            logger.info(
                "Saved evolution proposal",
                proposal_id=prop.id,
                type=prop.type
            )
