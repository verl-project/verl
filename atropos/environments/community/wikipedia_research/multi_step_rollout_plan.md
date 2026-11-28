# Wikipedia Article Creator: Multi-Step Tool-Calling Server Implementation Plan

## 1. Overview

This document outlines the plan for implementing a multi-step tool-calling environment in `environments/hack0/wikipedia/`. The environment will support Tavily search and webpage extraction tools, allowing an LLM to perform web research on a given topic and produce a comprehensive Wikipedia-style article through multiple interaction steps.

## 2. Key Components

### 2.1 State Management

The environment will maintain state across multiple interaction steps through an `EpisodeState` class:

```python
class EpisodeState:
    def __init__(self, episode_id: int, topic: str):
        self.episode_id = episode_id
        self.topic = topic  # The research topic for this episode
        self.message_history: List[Dict] = []  # Stores all interactions
        self.tool_calls: List[Dict] = []  # Records tool calls made
        self.tool_results: List[Dict] = []  # Records tool results returned
        self.steps_taken: int = 0  # Number of steps in this episode
        self.is_terminal: bool = False  # Whether episode has terminated
        self.final_article: Optional[str] = None  # Final Wikipedia article in markdown
        self.research_facts: List[str] = []  # Important facts discovered during research
        self.score: float = 0.0  # Score for this episode
```

### 2.2 Tool Integration

The environment will integrate Tavily tools:

1. **Web Search Tool**: Allows searching for information about the topic
   - Input: query, num_results (optional), filter_year (optional)
   - Output: Array of search results

2. **Page Extract Tool**: Allows extracting content from specific URLs for deep research
   - Input: url
   - Output: Object with title, content, success status

Tool definitions will be presented to the model in the system prompt with usage examples.

### 2.3 Multi-Step Research Flow

The interaction flow will follow this pattern:

1. Environment presents a topic to research (e.g., "Anti-black racism in the Arab World")
2. Model plans and conducts research using web_search and visit_page tools
3. Model gathers relevant information across multiple searches and page visits
4. When sufficient research is completed, model signals completion with "Final Step: ```markdown [article] ```"
5. Final Wikipedia-style article is evaluated for quality, comprehensiveness, and accuracy

### 2.4 Prompt Structure

System prompt will include:
- Task description for creating Wikipedia-style articles through research
- Wikipedia article style guidelines (NPOV, structure, citation style)
- Tool descriptions and usage examples
- Format for reasoning (`<think>...</think>`)
- Format for tool calls (`<tool_call>...</tool_call>`)
- Instructions for signaling completion (`Final Step: ```markdown [article] ````)

## 3. Implementation Details

### 3.1 `WikipediaArticleCreatorEnv` Class

This class will extend `BaseEnv` and implement:

```python
class WikipediaArticleCreatorEnv(BaseEnv):
    def __init__(self, config, server_configs, slurm=True, testing=False):
        # Initialize environment, tools, and tracking metrics

    async def _execute_tool_call(self, tool_call: Dict) -> Dict:
        # Execute a tool call and return the result

    def _parse_tool_calls(self, response: str) -> List[Dict]:
        # Extract tool calls from model response

    def _extract_final_article(self, response: str) -> Optional[str]:
        # Extract final Wikipedia article markdown if present

    async def _next_step(self, episode: EpisodeState) -> Tuple[bool, Dict]:
        # Process one step of article research interaction
        # Return (is_terminal, step_data)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        # Manage full research trajectory collection

    async def score(self, rollout_group_data) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        # Score model outputs based on article quality

    async def evaluate(self):
        # Run evaluation on test set of topics
```

### 3.2 Tool Execution Loop

The core multi-step research logic will be:

```python
async def _next_step(self, episode: EpisodeState) -> Tuple[bool, Dict]:
    # Get current conversation history
    messages = episode.message_history.copy()

    # Generate model response
    response = await self._get_model_response(messages)

    # Check for final article
    final_article = self._extract_final_article(response)
    if final_article:
        episode.is_terminal = True
        episode.final_article = final_article
        # Add response to history
        episode.message_history.append({"role": "assistant", "content": response})
        return True, {"response": response, "tool_calls": [], "tool_results": []}

    # Extract tool calls for research
    tool_calls = self._parse_tool_calls(response)

    # Execute research tool calls
    tool_results = []
    for tool_call in tool_calls:
        result = await self._execute_tool_call(tool_call)
        tool_results.append(result)

    # Add response and tool results to history
    episode.message_history.append({"role": "assistant", "content": response})

    # Format tool results as a user message
    tool_results_message = self._format_tool_results(tool_results)
    episode.message_history.append({"role": "user", "content": tool_results_message})

    # Update episode state
    episode.steps_taken += 1
    episode.tool_calls.extend(tool_calls)
    episode.tool_results.extend(tool_results)

    # Extract and store research facts for later evaluation
    self._extract_research_facts(tool_results, episode.research_facts)

    # Check if max steps reached
    if episode.steps_taken >= self.config.max_steps:
        episode.is_terminal = True

    return episode.is_terminal, {
        "response": response,
        "tool_calls": tool_calls,
        "tool_results": tool_results
    }
```

### 3.3 Article Scoring Mechanism

The scoring will evaluate the Wikipedia-style articles on:

1. **Research Thoroughness**: How well the model researches the topic
   - Diversity of sources consulted
   - Depth of investigation
   - Appropriate fact-finding

2. **Article Quality**: How well the final article adheres to Wikipedia standards
   - Neutral point of view (NPOV)
   - Comprehensive coverage of the topic
   - Proper structure (intro, sections, references)
   - Factual accuracy
   - Appropriate citation

3. **Research Efficiency**: How efficiently the model conducts research
   - Strategic search queries
   - Appropriate page selection for deeper reading
   - Minimal redundant searches

## 4. Configuration

The environment will be configurable with parameters:

```python
class WikipediaArticleCreatorConfig(BaseEnvConfig):
    max_steps: int = 10  # Maximum research steps per article
    temperature: float = 0.7  # Sampling temperature
    thinking_active: bool = True  # Enable thinking tags
    eval_topics: int = 30  # Number of topics for evaluation
    tool_timeout: float = 15.0  # Timeout for tool execution (seconds)
    tavily_api_key: Optional[str] = None  # API key for Tavily
    min_article_sections: int = 3  # Minimum number of sections in final article
    max_article_tokens: int = 2048  # Maximum tokens in final article
```

## 5. Example Interaction

Here's an example of the interaction flow for researching an article about "Anti-black racism in the Arab World":

**Initial Topic:**
```
Research and write a comprehensive Wikipedia-style article about: "Anti-black racism in the Arab World"
```

**Step 1 - Model Response:**
```
<think>
I need to research anti-black racism in the Arab world thoroughly. I should start with a general search to understand the scope and history of this issue.
</think>

I'll research this topic to create a Wikipedia-style article. Let me start with a general search.

<tool_call>
{"name": "web_search", "arguments": {"query": "history of anti-black racism in Arab world", "num_results": 5}}
</tool_call>
```

**Environment - Tool Results:**
```
Tool Results:
web_search(query="history of anti-black racism in Arab world", num_results=5)

[
  {
    "title": "Anti-Black Racism in the Arab World - Middle East Institute",
    "url": "https://www.mei.edu/publications/anti-black-racism-arab-world",
    "content": "Anti-Black racism exists across the Arab world and manifests in various discriminatory practices. This racism has historical roots in the Arab slave trade, which predated European slavery and continued into the 20th century in some regions...",
    "snippet": "Anti-Black racism exists across the Arab world and manifests in various discriminatory practices...",
    "date": "2020-06-15"
  },
  ...
]
```

**Step 2 - Model Response:**
```
<think>
This initial search provides useful context about the historical roots and modern manifestations of anti-black racism in the Arab world. I should now look more specifically at the historical Arab slave trade as it seems to be a key historical factor.
</think>

I need more specific information about the historical Arab slave trade to understand its connection to modern racism.

<tool_call>
{"name": "web_search", "arguments": {"query": "Arab slave trade East Africa history impact", "num_results": 5}}
</tool_call>
```

**Environment - Tool Results:**
```
Tool Results:
web_search(query="Arab slave trade East Africa history impact", num_results=5)

[
  {
    "title": "The Arab Slave Trade: 1,400 Years of History",
    "url": "https://www.fairplanet.org/story/the-arab-slave-trade-a-forgotten-crime-against-humanity/",
    "content": "The Arab slave trade began in the 7th century and lasted until the late 19th and early 20th centuries. The trade primarily involved the enslaving of Africans from the Sahel and East Africa regions...",
    "snippet": "The Arab slave trade began in the 7th century and lasted until the late 19th and early 20th centuries...",
    "date": "2021-02-21"
  },
  ...
]
```

**Step 3 - Model Response:**
```
<think>
I now have information about the historical context. To make this article comprehensive, I should also research contemporary manifestations of anti-black racism in different Arab countries.
</think>

Now I'll look for information about contemporary anti-black racism in specific Arab countries.

<tool_call>
{"name": "web_search", "arguments": {"query": "anti-black racism Egypt Lebanon Saudi Arabia contemporary", "num_results": 5}}
</tool_call>
```

**Environment - Tool Results:**
```
Tool Results:
web_search(query="anti-black racism Egypt Lebanon Saudi Arabia contemporary", num_results=5)

[
  {
    "title": "Confronting Anti-Black Racism in the Arab World",
    "url": "https://www.aljazeera.com/opinions/2020/6/15/confronting-anti-black-racism-in-the-arab-world",
    "content": "In Lebanon, Egypt, and other Arab countries, dark-skinned people, whether citizens or migrants, face various forms of discrimination. In Lebanon, domestic workers from Ethiopia and other African countries often face mistreatment under the kafala system...",
    "snippet": "In Lebanon, Egypt, and other Arab countries, dark-skinned people, whether citizens or migrants, face various forms of discrimination...",
    "date": "2020-06-15"
  },
  ...
]
```

**Step 4 - Model Response:**
```
<think>
I should look at some specific academic resources to ensure this article has scholarly backing. Let me search for academic articles on this topic.
</think>

I'd like to explore some academic perspectives on this issue.

<tool_call>
{"name": "web_search", "arguments": {"query": "scholarly articles anti-black racism Arab world", "num_results": 3}}
</tool_call>
```

**Environment - Tool Results:**
```
Tool Results:
web_search(query="scholarly articles anti-black racism Arab world", num_results=3)

[
  {
    "title": "Race and Racism in the Middle East and North Africa: Historical and Contemporary Perspectives",
    "url": "https://journals.sagepub.com/doi/full/10.1177/0021909620940465",
    "content": "This academic article examines how concepts of race and racial discrimination have evolved in MENA societies, with particular attention to anti-Black racism. The paper discusses how colonial influences interacted with pre-existing social hierarchies...",
    "snippet": "This academic article examines how concepts of race and racial discrimination have evolved in MENA societies...",
    "date": "2020-07-28"
  },
  ...
]
```

**Step 5 - Model Response:**
```
<think>
I should find information about resistance movements and efforts to combat anti-black racism in the Arab world to provide a balanced perspective.
</think>

Let me research efforts to combat anti-black racism in the Arab world.

<tool_call>
{"name": "web_search", "arguments": {"query": "anti-racism movements Arab world Black Lives Matter influence", "num_results": 3}}
</tool_call>
```

**Environment - Tool Results:**
```
Tool Results:
web_search(query="anti-racism movements Arab world Black Lives Matter influence", num_results=3)

[
  {
    "title": "Black Lives Matter Reaches Middle East and North Africa",
    "url": "https://institute.global/policy/black-lives-matter-reaches-middle-east-and-north-africa",
    "content": "The Black Lives Matter movement has inspired activists across the Middle East and North Africa to confront anti-Black racism in their societies. In Tunisia, Lebanon, and other countries, new anti-racism initiatives have emerged...",
    "snippet": "The Black Lives Matter movement has inspired activists across the Middle East and North Africa...",
    "date": "2020-07-13"
  },
  ...
]
```

**Final Step - Model Response:**
```
<think>
I now have gathered substantial information covering the historical context, contemporary manifestations, academic perspectives, and resistance efforts regarding anti-black racism in the Arab world. I can now write a comprehensive Wikipedia-style article on this topic, organizing the information into appropriate sections.
</think>

Final Step: ```markdown
# Anti-Black Racism in the Arab World

Anti-Black racism in the Arab world refers to prejudice, discrimination, and systemic bias against people of African descent across Arab countries. This phenomenon has deep historical roots and continues to manifest in various forms in contemporary Arab societies. Despite growing awareness and resistance movements, anti-Black racism remains a significant social issue throughout the region.

## Historical Context

### The Arab Slave Trade

The historical Arab slave trade, which began in the 7th century and continued until the early 20th century in some regions, plays a significant role in understanding the origins of anti-Black racism in the Arab world. Unlike the transatlantic slave trade, the Arab slave trade was not exclusively focused on Africans, though they constituted a major portion of those enslaved.

The trade primarily involved the capture and transportation of people from East Africa, the Sahel region, and the Horn of Africa to destinations across the Middle East, North Africa, and parts of Asia. It is estimated that between 10-18 million Africans were enslaved during this period, although precise figures remain disputed among historians.

Enslaved Africans in the Arab world were typically employed as domestic servants, agricultural laborers, military personnel (including the famous slave-soldiers or "Mamluks"), and concubines. The practice created lasting racial hierarchies that influenced social structures across the region.

### Colonial Influence and Racial Hierarchies

European colonial powers introduced or reinforced racial categorizations throughout the region during the 19th and 20th centuries. This colonial influence often intersected with pre-existing social hierarchies, further cementing anti-Black attitudes in many Arab societies.

## Contemporary Manifestations

### Social Discrimination

In contemporary Arab societies, anti-Black racism manifests in various forms of social discrimination:

* Pejorative language and racial slurs targeting dark-skinned individuals
* Negative media stereotypes and underrepresentation
* Discriminatory beauty standards that privilege lighter skin
* Social segregation and barriers to intermarriage with darker-skinned communities

### Economic and Labor Issues

Economic discrimination against Black communities and individuals is evident across multiple Arab countries:

* In Lebanon, the kafala sponsorship system has been particularly criticized for enabling the exploitation of African migrant workers, especially women from Ethiopia, Kenya, and other East African nations
* In Gulf countries, labor hierarchies often place Black migrant workers at the bottom, with lower wages and poorer working conditions
* Limited economic mobility for Black citizens in countries like Iraq, Egypt, and Mauritania

### Legal Status and Citizenship

In several Arab countries, Black communities face challenges related to legal recognition and citizenship rights:

* In countries like Kuwait and Iraq, some Black descendants of former slaves continue to face difficulties obtaining full citizenship rights
* Black refugees and migrants often encounter greater obstacles in obtaining legal status
* Discriminatory policing and law enforcement practices affect Black communities disproportionately

## Black Communities in the Arab World

### Indigenous Black Populations

Several Arab countries have indigenous Black populations with deep historical roots:

* The Nubians in Egypt and Sudan
* The Afro-Iraqi community, many descended from slaves brought to Iraq during the Abbasid Caliphate
* The Haratin in Morocco and Mauritania
* Black Bedouin tribes in parts of the Levant

These communities often face marginalization while simultaneously contributing significantly to local cultures.

### Migrant Communities

More recent African migrant communities exist across the Arab world, particularly in:

* Gulf countries, where large numbers of East and West African workers reside
* Lebanon and Jordan, which host various African migrant communities
* Libya, which has been both a destination and transit point for sub-Saharan African migrants

## Combating Anti-Black Racism

### Legal Reforms

Several Arab countries have enacted legislation against racial discrimination in recent years:

* Tunisia passed landmark anti-racism legislation in 2018, making it the first Arab country with a specific law criminalizing racial discrimination
* Morocco and other countries have reformed labor laws to better protect migrant workers

### Activism and Awareness

Anti-racism activism has grown across the Arab world, particularly following the global Black Lives Matter movement:

* Social media campaigns like #BlackLivesMatter_MENA have raised awareness about anti-Black racism
* Organizations such as Mnemty in Tunisia and Anbar in Lebanon advocate for racial equality
* Cultural initiatives highlighting Afro-Arab heritage challenge negative stereotypes

### Religious Perspectives

Islamic scholars and religious leaders have increasingly emphasized Islamic teachings against racism:

* Reference to the Prophet Muhammad's statements against racial discrimination
* Religious discourse highlighting racial equality in Islam
* Interreligious initiatives promoting racial harmony

## Media Representation and Cultural Progress

Recent years have seen some progress in the representation of Black Arabs in media and culture:

* Increased visibility of Black actors, musicians, and public figures in Arab media
* Literary and artistic works addressing themes of racism and racial identity
* Academic research and publications examining racism in Arab societies

## Challenges and Future Directions

Despite progress, significant challenges remain in addressing anti-Black racism in the Arab world:

* Limited acknowledgment of historical slavery and its legacy
* Persistent colorism and beauty standards favoring lighter skin
* Economic inequalities that disproportionately affect Black communities
* Need for more comprehensive anti-discrimination legislation across the region

Scholars and activists emphasize that confronting anti-Black racism requires addressing both historical legacies and contemporary forms of discrimination through education, legal reform, and cultural change.

## References

1. Middle East Institute. (2020). "Anti-Black Racism in the Arab World."
2. Fair Planet. (2021). "The Arab Slave Trade: 1,400 Years of History."
3. Al Jazeera. (2020). "Confronting Anti-Black Racism in the Arab World."
4. Journal of International Studies. (2020). "Race and Racism in the Middle East and North Africa: Historical and Contemporary Perspectives."
5. Tony Blair Institute for Global Change. (2020). "Black Lives Matter Reaches Middle East and North Africa."
```

## 6. Implementation Timeline

1. **Phase 1**: Create core EpisodeState and WikipediaArticleCreatorEnv classes
2. **Phase 2**: Implement multi-step research interaction logic
3. **Phase 3**: Develop article evaluation metrics and scoring
4. **Phase 4**: Fine-tune Wikipedia-style prompting
5. **Phase 5**: Create test topic dataset and optimize

## 7. Key Challenges and Solutions

1. **Challenge**: Maintaining Wikipedia Neutral Point of View (NPOV)
   **Solution**: Clear guidelines in prompt, evaluation metrics for neutrality

2. **Challenge**: Research depth vs. time constraints
   **Solution**: Strategic search strategies, efficient information extraction

3. **Challenge**: Balancing comprehensiveness with readability
   **Solution**: Clear article structure guidelines with emphasis on hierarchical organization

4. **Challenge**: Verifying factual accuracy
   **Solution**: Multiple source verification, research fact tracking

5. **Challenge**: Managing context length for complex topics
   **Solution**: Implement hierarchical summarization techniques

## 8. Metrics for Success

1. **Research Quality**:
   - Source diversity: Number of unique, high-quality sources consulted
   - Query strategy: Effectiveness of search queries in discovering relevant information
   - Fact discovery rate: Number of key facts uncovered per search

2. **Article Quality**:
   - Structure adherence: Proper Wikipedia format with introduction, sections, references
   - NPOV compliance: Balanced presentation of different viewpoints
   - Factual accuracy: Percentage of statements that are verifiably correct
   - Comprehensiveness: Coverage of major aspects of the topic
   - Citation quality: Proper attribution of information to reliable sources

3. **Efficiency Metrics**:
   - Steps to completion: Number of research steps needed
   - Information utility ratio: Useful information extracted per tool call

## 9. Next Steps

1. Implement core `WikipediaArticleCreatorEnv` class
2. Create a database of diverse research topics
3. Develop Wikipedia-style article evaluation metrics
4. Build comprehensive prompt templates with Wikipedia guidelines
5. Create testing infrastructure for article quality assessment
