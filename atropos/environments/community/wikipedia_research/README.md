# Wikipedia Article Research Environment

This environment trains LLMs to research and create Wikipedia-style articles on arbitrary topics using web search and content extraction tools.

## Overview

The Wikipedia Article Research Environment provides a comprehensive framework for training language models to conduct multi-step research and generate high-quality, factually accurate Wikipedia-style articles. The environment combines web search capabilities with content extraction tools to enable thorough research processes.

## Core Features

### Multi-Step Research Process
- **Web Search Integration**: Uses Tavily API for comprehensive web search capabilities
- **Content Extraction**: Extracts full content from specific webpages for detailed analysis
- **Research Fact Tracking**: Automatically tracks and stores important facts discovered during research
- **Wikipedia Blocking**: Prevents direct access to Wikipedia to encourage diverse source usage

### Article Quality Assessment
- **Structure Scoring**: Evaluates article organization, section structure, and references
- **Comprehensiveness Scoring**: Assesses coverage of important topic aspects
- **Fact Usage Scoring**: Measures effective incorporation of researched facts
- **Factual Accuracy Evaluation**: Optional OpenAI-powered line-by-line fact-checking against reference articles

### Advanced Evaluation System
- **Dual Scoring Mechanisms**: Combines structural quality with factual accuracy
- **Line-by-Line Analysis**: Categorizes statements as CORRECT, INCORRECT, or UNKNOWN
- **Reference Comparison**: Compares generated articles against real Wikipedia content
- **Comprehensive Metrics**: Provides detailed accuracy statistics and combined scores

## Technical Implementation

### Environment Configuration
- **Environment Name**: `WikipediaArticleCreator`
- **Base Class**: `BaseEnv` from atroposlib
- **Tool Integration**: Tavily search and extraction tools
- **Evaluation**: OpenAI-powered factual accuracy assessment

### Key Components
- **Episode Management**: Tracks research sessions with conversation history
- **Tool Execution**: Handles web search and content extraction with error handling
- **Quality Metrics**: Multi-dimensional article assessment framework
- **W&B Integration**: Comprehensive logging and visualization support

### Research Tools
1. **Web Search** (`web_search`): Searches the web with configurable result limits and year filtering
2. **Page Extraction** (`visit_page`): Extracts content from specific URLs with error handling

## Setup and Configuration

### Environment Variables
```bash
# Required for web research
export TAVILY_API_KEY="your_tavily_api_key"

# Required for LLM access
export OPENAI_API_KEY="your_openai_api_key"

# Optional configuration
export MODEL_NAME="gpt-4o"
export MAX_STEPS="10"
export TEMPERATURE="0.7"
```

### Dependencies
```bash
pip install openai tavily-python python-dotenv smolagents pandas pyyaml
```

## Usage Examples

### Training Mode
```bash
python -m atroposlib.cli.dpo \
    --env-module "environments.community.wikipedia_research.wikipedia_article_creator" \
    --wandb-mode online
```

### Evaluation Mode
```bash
python -m atroposlib.cli.sft \
    --eval-only \
    --env-module "environments.community.wikipedia_research.wikipedia_article_creator"
```

### Direct Usage
```bash
cd environments/community/wikipedia_research
python run_with_openai.py --topic "Climate change in Antarctica" --model "gpt-4o" --max-steps 10
```

## Evaluation Metrics

### Quality Metrics (0-1 scale)
- **Structure Score**: Article organization and section quality
- **Comprehensiveness**: Coverage of important topic aspects
- **Fact Usage**: Effective incorporation of researched information
- **Overall Quality**: Combined structural and content quality

### Factual Accuracy Metrics
- **Correct Statements**: Percentage verified as factually accurate
- **Incorrect Statements**: Percentage contradicting reference sources
- **Unknown Statements**: Percentage that cannot be verified
- **Accuracy Score**: Net accuracy in [-1, 1] range

### Combined Metrics
- **Overall Article Score**: Comprehensive quality + accuracy metric in [-1, 1] range
- **Research Efficiency**: Steps taken vs. article quality achieved
- **Tool Usage Effectiveness**: Success rate of research tool calls

## Configuration Parameters

### Core Settings
- `max_steps`: Maximum research steps per article (default: 10)
- `temperature`: Sampling temperature for generation (default: 0.7)
- `eval_topics`: Number of topics for evaluation (default: 30)
- `tool_timeout`: Timeout for tool execution in seconds (default: 15.0)

### Quality Thresholds
- `min_article_sections`: Minimum sections required (default: 3)
- `max_article_tokens`: Maximum article length (default: 2048)

### Advanced Options
- `thinking_active`: Enable reasoning tags (default: True)
- `logging_active`: Enable detailed logging (default: True)
- `include_messages`: Include conversation history in outputs (default: True)

## Research Workflow

1. **Topic Assignment**: Model receives a research topic
2. **Research Planning**: Model develops research strategy using `<think>` tags
3. **Information Gathering**: Uses `web_search` and `visit_page` tools iteratively
4. **Fact Extraction**: Environment tracks important facts from tool results
5. **Article Generation**: Model synthesizes research into Wikipedia-style article
6. **Quality Assessment**: Environment evaluates structure, comprehensiveness, and accuracy
7. **Factual Verification**: Optional comparison against reference Wikipedia articles

## Output Format

### Article Structure
Articles must be formatted as:
```
Final Step: ```markdown
# Article Title

## Introduction
[Content...]

## Section 1
[Content...]

## References
[Sources...]
```

### Tool Call Format
```xml
<tool_call>
{"name": "web_search", "arguments": {"query": "search terms", "num_results": 5}}
</tool_call>

<tool_call>
{"name": "visit_page", "arguments": {"url": "https://example.com"}}
</tool_call>
```

## Performance Characteristics

### Computational Requirements
- **Memory**: ~1-2 GB RAM for typical usage
- **API Calls**: 10-50 tool calls per article depending on complexity
- **Processing Time**: 2-10 minutes per article with OpenAI models
- **Storage**: Minimal local storage requirements

### Scalability
- **Concurrent Episodes**: Supports multiple parallel research sessions
- **Batch Processing**: Configurable batch sizes for training
- **Tool Rate Limiting**: Built-in respect for API rate limits
- **Error Recovery**: Robust error handling for network issues

## Integration Features

### W&B Logging
- **Conversation Tracking**: Complete research session histories
- **Quality Metrics**: Detailed article assessment data
- **Tool Usage Analytics**: Search and extraction success rates
- **Accuracy Statistics**: Factual verification results

### HTML Rendering
- **Research Visualization**: Complete conversation flows with tool results
- **Article Presentation**: Formatted final articles with metadata
- **Quality Dashboards**: Interactive metric displays

This environment provides a comprehensive framework for training LLMs to conduct thorough research and generate high-quality, factually accurate articles while maintaining transparency in the research process.
