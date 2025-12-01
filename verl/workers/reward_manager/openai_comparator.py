"""
OpenAI GPT-5 mini Comparator for Pairwise Comparison  
使用OpenAI GPT-5 mini进行两两比较的纯净实现
"""

from typing import Optional, Dict, Any
import json
import os
import time
import asyncio

class OpenAIComparator:
    """使用自定义推理模型进行两两比较的比较器"""
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "rioy/reasoning-model-step60-1e-5:de-32ff8d40be11e37b",
                 api_base: str = "https://api.novita.ai/dedicated/v1/openai",
                 max_retries: int = 3,
                 max_concurrent: int = 10):
        """
        初始化自定义推理模型比较器
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            api_base: API基础URL
            max_retries: API调用失败时的重试次数
            max_concurrent: 最大并发调用数量
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.api_key = api_key
        self.openai_client = self._init_openai_client(self.api_key, self.api_base)
        # 创建异步客户端
        self.async_openai_client = self._init_async_openai_client(self.api_key, self.api_base)
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        print(f"🚀 Initializing custom reasoning model comparator: {self.model_name}")
        
    def _init_openai_client(self, api_key: str, api_base: str):
        """初始化同步客户端"""
        try:
            from openai import OpenAI
            
            if not api_key:
                raise ValueError("API key not provided.")
            
            client = OpenAI(api_key=api_key, base_url=api_base)
            print(f"✅ Custom model sync client initialized with model: {self.model_name}")
            return client
            
        except ImportError:
            print("❌ OpenAI package not installed. Install with: pip install openai")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI sync client: {e}")
            return None

    def _init_async_openai_client(self, api_key: str, api_base: str):
        """初始化异步客户端"""
        try:
            from openai import AsyncOpenAI
            
            if not api_key:
                print("❌ No API key provided for async client")
                return None
            
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            print(f"✅ Custom model async client initialized with model: {self.model_name}")
            return client
            
        except ImportError:
            print("❌ OpenAI package not installed. Install with: pip install openai")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI async client: {e}")
            return None
    
    def compare_responses(self, prompt: str, response1: str, response2: str) -> int:
        """
        使用GPT-5 mini比较两个回答
        
        Args:
            prompt: 原始问题
            response1: 第一个回答
            response2: 第二个回答
            
        Returns:
            int: 0=tie, 1=response1更好, 2=response2更好
        """
        if self.openai_client is None:
            print("⚠️ OpenAI client not available, using fallback (tie)")
            return 0
        
        comparison_prompt = self._build_comparison_prompt(prompt, response1, response2)
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": comparison_prompt
                        }
                    ],
                    max_tokens=4096,
                    temperature=1,
                    top_p=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                    timeout=600
                )
                
                result_text = response.choices[0].message.content
                winner = self._parse_openai_result(result_text)
                
                return winner
                
            except Exception as e:
                print(f"❌ OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    
        # 所有重试都失败了，返回平局避免训练中断
        print("⚠️ All OpenAI API attempts failed, using fallback (tie) to continue training")
        return 0

    async def compare_responses_async(self, prompt: str, response1: str, response2: str) -> int:
        """
        异步比较两个回答
        
        Args:
            prompt: 原始问题
            response1: 第一个回答  
            response2: 第二个回答
            
        Returns:
            int: 0=tie, 1=response1更好, 2=response2更好
        """
        if self.async_openai_client is None:
            print("⚠️ Async OpenAI client not available, using fallback (tie)")
            return 0
        
        async with self.semaphore:  # 控制并发数
            comparison_prompt = self._build_comparison_prompt(prompt, response1, response2)
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.async_openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user", 
                                "content": comparison_prompt
                            }
                        ],
                        max_tokens=4096,
                        temperature=1,
                        top_p=1,
                        presence_penalty=0,
                        frequency_penalty=0,
                        timeout=600
                    )
                    
                    result_text = response.choices[0].message.content
                    winner = self._parse_openai_result(result_text)
                    
                    return winner
                    
                except Exception as e:
                    print(f"❌ OpenAI async API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        
            # 所有重试都失败了，返回平局避免训练中断
            print("⚠️ All async OpenAI API attempts failed, using fallback (tie) to continue training")
            return 0
    
    def compare_responses_with_details(self, prompt: str, response1: str, response2: str) -> dict:
        """
        使用GPT-5 mini比较两个回答并返回详细信息
        
        Returns:
            dict: {
                "winner": int,
                "prompt": str,
                "response1": str, 
                "response2": str,
                "ai_response": str,
                "error_msg": str (if any)
            }
        """
        details = {
            "prompt": prompt,
            "response1": response1,
            "response2": response2,
            "ai_response": "",
            "error_msg": "",
            "winner": 0
        }
        
        if self.openai_client is None:
            details["error_msg"] = "OpenAI client not available"
            details["winner"] = 0
            return details
        
        comparison_prompt = self._build_comparison_prompt(prompt, response1, response2)
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": comparison_prompt
                        }
                    ],
                    max_tokens=4096,
                    temperature=1,
                    top_p=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                    timeout=600
                )
                
                result_text = response.choices[0].message.content
                details["ai_response"] = result_text
                winner = self._parse_openai_result(result_text)
                details["winner"] = winner
                
                return details
                
            except Exception as e:
                details["error_msg"] = f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        # 所有重试都失败了
        details["error_msg"] += " - All attempts failed, using fallback (tie)"
        details["winner"] = 0
        return details

    async def compare_responses_with_details_async(self, prompt: str, response1: str, response2: str) -> dict:
        """
        异步比较两个回答并返回详细信息
        
        Returns:
            dict: {
                "winner": int,
                "prompt": str,
                "response1": str, 
                "response2": str,
                "ai_response": str,
                "error_msg": str (if any)
            }
        """
        details = {
            "prompt": prompt,
            "response1": response1,
            "response2": response2,
            "ai_response": "",
            "error_msg": "",
            "winner": 0
        }
        
        if self.async_openai_client is None:
            details["error_msg"] = "Async OpenAI client not available"
            details["winner"] = 0
            return details
        
        async with self.semaphore:  # 控制并发数
            comparison_prompt = self._build_comparison_prompt(prompt, response1, response2)
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.async_openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user", 
                                "content": comparison_prompt
                            }
                        ],
                        max_tokens=4096,
                        temperature=1,
                        top_p=1,
                        presence_penalty=0,
                        frequency_penalty=0,
                        timeout=600
                    )
                    
                    result_text = response.choices[0].message.content
                    details["ai_response"] = result_text
                    winner = self._parse_openai_result(result_text)
                    details["winner"] = winner
                    
                    return details
                    
                except Exception as e:
                    details["error_msg"] = f"OpenAI async API error (attempt {attempt + 1}/{self.max_retries}): {e}"
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        
            # 所有重试都失败了
            details["error_msg"] += " - All attempts failed, using fallback (tie)"
            details["winner"] = 0
            return details
    
    def _build_comparison_prompt(self, prompt: str, response1: str, response2: str) -> str:
        """构造UI设计比较prompt"""
        return f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below.

[Client Question]
{prompt}

[The Start of Chatbot A's Response]
{response1}

[The End of Chatbot A's Response]

[The Start of Chatbot B's Response]
{response2}

[The End of Chatbot B's Response]

Output your final verdict at last by strictly following this format: '<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better."""
    
    def _parse_openai_result(self, result_text: str) -> int:
        """
        解析自定义模型的比较结果
        
        Args:
            result_text: 模型返回的文本，期望格式: <answer>[[A]]</answer> 或 <answer>[[B]]</answer>
            
        Returns:
            int: 0=tie, 1=response1更好, 2=response2更好
        """
        try:
            # 检查空响应
            if not result_text or result_text.strip() == "":
                print("⚠️ Custom model returned empty response, using fallback (tie)")
                return 0
            
            # 使用正则表达式提取答案
            import re
            pattern = r'<answer>\[\[([AB])\]\]</answer>'
            match = re.search(pattern, result_text)
            
            if match:
                winner = match.group(1).upper()
                if winner == "A":
                    return 1  # response1更好
                elif winner == "B":
                    return 2  # response2更好
            
            # 如果没有找到标准格式，尝试更多的解析模式
            loose_patterns = [
                r'\[\[([AB])\]\]',  # 直接匹配 [[A]] 或 [[B]]
                r'answer[:\s]*([AB])',  # answer: A 或 answer B  
                r'选择[:\s]*([AB])',  # 选择: A 或 选择 A
                r'答案[:\s]*([AB])',  # 答案: A 或 答案 A
                r'winner[:\s]*([AB])',  # winner: A 或 winner B
                r'better[:\s]*([AB])',  # better: A 或 better B
            ]
            
            for pattern in loose_patterns:
                match = re.search(pattern, result_text, re.IGNORECASE)
                if match:
                    winner = match.group(1).upper()
                    if winner == "A":
                        return 1
                    elif winner == "B":
                        return 2
            
            print(f"⚠️ Custom model returned invalid format. Raw response: {result_text[:150]}... Using fallback (tie)")
            return 0
            
        except Exception as e:
            print(f"⚠️ Unexpected error parsing custom model result: {e}. Raw response: {result_text[:150]}... Using fallback (tie)")
            return 0
    


class CustomModelPairwiseComparator:
    """专门使用自定义推理模型的配对比较器"""
    
    def __init__(self, **kwargs):
        """
        初始化自定义推理模型比较器
        
        Args:
            **kwargs: 传递给OpenAIComparator的参数
        """
        self.openai_comparator = OpenAIComparator(**kwargs)
    
    def compare(self, prompt: str, response1: str, response2: str) -> dict:
        """
        执行比较并返回winner和详细信息（同步版本）
        
        Returns:
            dict: {
                "winner": int (0=tie, 1=response1, 2=response2),
                "details": dict - 详细的比较信息
            }
        """
        details = self.openai_comparator.compare_responses_with_details(prompt, response1, response2)
        
        return {
            "winner": details["winner"],
            "details": details
        }

    async def compare_async(self, prompt: str, response1: str, response2: str) -> dict:
        """
        执行异步比较并返回winner和详细信息
        
        Returns:
            dict: {
                "winner": int (0=tie, 1=response1, 2=response2),
                "details": dict - 详细的比较信息
            }
        """
        details = await self.openai_comparator.compare_responses_with_details_async(prompt, response1, response2)
        
        return {
            "winner": details["winner"],
            "details": details
        }
