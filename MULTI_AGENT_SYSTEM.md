# Multi-Agent System Documentation

## Overview

The enhanced agent system now generates multiple agents based on both explicit `agent_type` parameters and contextual analysis of the message content. This allows for sophisticated workflows where different specialized agents work together to complete complex tasks.

## Key Features

### 1. **Dual Agent Generation**
- **Primary Agent**: Generated from explicit `agent_type` parameter or detected from message content
- **Additional Agents**: Detected from message context using keyword analysis and LLM reasoning

### 2. **Agent Chaining**
- Agents execute in sequence, passing results between each other
- Each agent receives context from previous agents when relevant
- Automatic PDF generation for analysis results when email is needed

### 3. **Intelligent Context Analysis**
- Keyword-based detection for common patterns
- LLM-powered context analysis for complex scenarios
- Support for multiple agent types in a single request

## Supported Agent Types

| Agent Type | Purpose | Tools Available |
|------------|---------|----------------|
| `analyzer` | Document analysis, financial statements | `analyze_financial_document`, `generate_analysis_pdf`, `detect_document_type` |
| `mailer` | Send emails with attachments | `send_email`, `send_email_with_pdf` |
| `summarizer` | Document summarization | `summarize_document` |
| `mathematical` | Calculations and math problems | `calculate_math` |
| `historian` | Historical information research | `search_historical_info` |
| `sports` | Sports information and statistics | `search_sports_info` |
| `general` | General assistance | No specific tools |

## Usage Examples

### Example 1: Financial Analysis + Email
```json
{
  "agent_type": "financial",
  "message": "analyze this balance sheet and send the results to john@company.com",
  "files": ["balance_sheet.pdf"]
}
```

**Result**: 
1. Creates `analyzer` agent (financial maps to analyzer)
2. Detects `mailer` agent from "send the results to john@company.com"
3. **Workflow**: analyzer → PDF generation → mailer

### Example 2: Context-Only Detection
```json
{
  "message": "calculate the ROI of this investment and email the results to the team",
  "files": []
}
```

**Result**:
1. Detects `mathematical` agent from "calculate the ROI"
2. Detects `mailer` agent from "email the results to the team"
3. **Workflow**: mathematical → mailer

### Example 3: Document Analysis Only
```json
{
  "agent_type": "analyzer",
  "message": "examine this quarterly report",
  "files": ["report.pdf"]
}
```

**Result**:
1. Uses explicit `analyzer` agent
2. No additional agents detected
3. **Workflow**: analyzer only

### Example 4: Complex Multi-Agent
```json
{
  "message": "summarize this sports report, calculate team statistics, and email to coach@team.com",
  "files": ["sports_report.pdf"]
}
```

**Result**:
1. Detects `summarizer` agent (primary from "summarize")
2. Detects `mathematical` agent from "calculate team statistics"
3. Detects `mailer` agent from "email to coach@team.com"
4. **Workflow**: summarizer → mathematical → mailer

## API Endpoints

### `/ask` (Enhanced)
- **Purpose**: General endpoint with backward compatibility
- **Behavior**: Now uses the multi-agent system internally
- **Parameters**: `message`, `agent_type` (optional), `files` (optional)

### `/multi-agent` (New)
- **Purpose**: Explicit multi-agent endpoint with detailed logging
- **Behavior**: Full visibility into agent detection and chaining process
- **Parameters**: `message`, `agent_type` (optional), `files` (optional)
- **Additional**: Returns detailed metadata about agent chain and execution

## Response Format

```json
{
  "workflow_type": "multi_agent",
  "agent_chain": ["analyzer", "mailer"],
  "agents_executed": ["analyzer", "mailer"],
  "primary_agent": "analyzer",
  "additional_agents": ["mailer"],
  "individual_results": {
    "analyzer": "Detailed financial analysis...",
    "mailer": "Email sent successfully to john@company.com"
  },
  "pdf_generated": "PDF report generated successfully: generated_reports/analysis_20240915_143022.pdf",
  "pdf_path": "generated_reports/analysis_20240915_143022.pdf",
  "response": "=== Analyzer Results ===\nDetailed analysis...\n\n=== Mailer Results ===\nEmail sent...",
  "uploaded_files": ["balance_sheet.pdf"]
}
```

## Context Detection Logic

### Keyword-Based Detection
The system scans messages for specific keywords associated with each agent type:

- **Mailer**: "send mail", "email", "send to", "share via email", "notify", "distribute"
- **Analyzer**: "analyze", "analysis", "examine", "review", "financial", "balance sheet"
- **Mathematical**: "calculate", "compute", "solve", "math", "+", "-", "*", "/"
- **Summarizer**: "summarize", "summary", "brief", "overview"
- **Historian**: "history", "historical", "ancient", "war", "empire"
- **Sports**: "sports", "athlete", "team", "game", "score", "championship"

### LLM-Powered Analysis
When keyword detection is insufficient, the system uses LLM reasoning to identify:
- Secondary tasks mentioned in the message
- Implicit requests for additional functionality
- Complex multi-step workflows

## Agent Chaining Logic

1. **Primary Agent Execution**: Runs first with original message and files
2. **Context Passing**: Results from primary agent are passed to subsequent agents
3. **PDF Generation**: Automatically generated for analysis results when email is needed
4. **Email Integration**: Mailer agents receive previous results and PDF paths
5. **Error Handling**: Individual agent failures don't stop the entire chain

## Implementation Details

### Key Functions

- `analyze_context_for_additional_agents()`: Detects additional agents needed
- `run_multi_agent_workflow()`: Orchestrates the entire multi-agent process
- `create_dynamic_agent()`: Creates specialized agents with appropriate tools
- `detect_agent_type()`: Determines primary agent type from message content

### Agent-Specific Context

Each agent receives tailored context based on its position in the chain:
- **Analyzer**: Instructions to focus only on analysis
- **Mailer**: Previous results and PDF paths for attachment
- **Summarizer**: Previous analysis results if available

## Benefits

1. **Intelligent Automation**: Single request can trigger complex multi-step workflows
2. **Context Awareness**: System understands implicit requests and secondary tasks
3. **Seamless Integration**: Agents work together without manual coordination
4. **Flexible Input**: Works with explicit agent types or pure message analysis
5. **Rich Output**: Detailed results from each agent plus combined workflow summary

## Future Enhancements

- Support for parallel agent execution where appropriate
- Custom agent type definitions
- Workflow templates for common patterns
- Integration with external services and APIs
- Advanced context sharing between agents
