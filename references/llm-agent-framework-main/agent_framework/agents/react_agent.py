"""
ReAct Agent Implementation

Implements the ReAct (Reasoning + Acting) pattern for agent decision making.
"""

import re
from typing import List, Dict, Any, Optional, Union
from agent_framework.agents.base_agent import BaseAgent, AgentAction, AgentFinish


class ReActAgent(BaseAgent):
    """ReAct agent that reasons and acts iteratively"""
    
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any], max_iterations: int = 10):
        super().__init__(name, description, llm, tools)
        self.max_iterations = max_iterations
    
    def _build_prompt(self, task: str, scratchpad: str = "") -> str:
        """Build ReAct prompt with tools and scratchpad"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""Answer the following question as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{", ".join(self.tools.keys())}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {task}
{scratchpad}"""
        
        return prompt
    
    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse LLM output into action or finish"""
        # Check for final answer
        if "Final Answer:" in text:
            return AgentFinish(
                output=text.split("Final Answer:")[-1].strip(),
                log=text
            )
        
        # Parse action
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            return AgentAction(
                tool=action,
                tool_input={"query": action_input},
                log=text
            )
        
        # If parsing fails, return a finish with the text
        return AgentFinish(output=text, log=text)
    
    def plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> List[AgentAction]:
        """Plan is integrated into run() for ReAct agents"""
        raise NotImplementedError("ReAct agents don't separate planning from execution")
    
    def execute(self, action: AgentAction) -> str:
        """Execute an action using the specified tool"""
        tool = self.get_tool(action.tool)
        if tool is None:
            return f"Error: Tool '{action.tool}' not found"
        
        try:
            result = tool.run(**action.tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def run(self, task: str) -> str:
        """Run the ReAct agent on a task"""
        scratchpad = ""
        
        for i in range(self.max_iterations):
            # Build prompt with current scratchpad
            prompt = self._build_prompt(task, scratchpad)
            
            # Get LLM response
            try:
                response = self.llm.generate(prompt)
            except Exception as e:
                return f"Error: LLM failed - {str(e)}"
            
            # Parse output
            output = self._parse_output(response)
            
            # Store in memory
            self.add_to_memory({
                'iteration': i,
                'prompt': prompt,
                'response': response,
                'output': output
            })
            
            # Check if finished
            if isinstance(output, AgentFinish):
                return output.output
            
            # Execute action
            observation = self.execute(output)
            
            # Update scratchpad
            scratchpad += f"Thought: {output.log}\n"
            scratchpad += f"Observation: {observation}\n"
        
        return f"Error: Reached maximum iterations ({self.max_iterations})"
