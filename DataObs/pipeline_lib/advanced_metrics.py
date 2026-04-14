"""
Advanced data metrics: text similarity, entropy, and difficulty assessment
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import math

logger = logging.getLogger(__name__)


# ============================================================================
# 0. DIVERSITY FRAMEWORK (统一框架，支持多种相似度函数)
# ============================================================================

from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class SimilarityType(Enum):
    """支持的相似度函数类型"""
    JACCARD = "jaccard"           # 词级 Jaccard
    LEVENSHTEIN = "levenshtein"   # 编辑距离
    COSINE = "cosine"             # TF-IDF 余弦
    JARO_WINKLER = "jaro_winkler" # Jaro-Winkler
    NGRAM = "ngram"               # N-gram 相似度
    BERTOUCH = "bertouch"         # BERT 语义相似度
    BLEU = "bleu"                 # BLEU score (词级)
    ROUGE = "rouge"               # ROUGE score


@dataclass
class DiversityConfig:
    """Diversity 计算配置"""
    similarity_type: SimilarityType = SimilarityType.JACCARD
    sample_size: int = 100
    ngram_n: int = 2              # N-gram 的 n 值
    bert_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # BERT 模型
    compute_on: str = "both"    # "prompt", "answer", "both"


class SimilarityFunction:
    """相似度函数基类"""

    def compute(self, text1: str, text2: str) -> float:
        """计算两个文本之间的相似度"""
        raise NotImplementedError

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        """计算文本对的相似度矩阵"""
        n = len(texts)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.compute(texts[i], texts[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix


class JaccardSimilarity(SimilarityFunction):
    """词级 Jaccard 相似度"""

    def compute(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


class LevenshteinSimilarity(SimilarityFunction):
    """编辑距离相似度 (1 - normalized distance)"""

    def compute(self, text1: str, text2: str) -> float:
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        dist = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        return 1 - (dist / max_len) if max_len > 0 else 1.0


class CosineSimilarity(SimilarityFunction):
    """TF-IDF 余弦相似度"""

    def __init__(self):
        self.vectorizer = None

    def compute(self, text1: str, text2: str) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer()
                self.vectorizer.fit([text1, text2])
            vec1 = self.vectorizer.transform([text1])
            vec2 = self.vectorizer.transform([text2])
            similarity = (vec1 * vec2.T).toarray()[0, 0]
            return float(similarity)
        except ImportError:
            logger.warning("sklearn not available")
            return JaccardSimilarity().compute(text1, text2)

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            return (tfidf_matrix * tfidf_matrix.T).toarray()
        except ImportError:
            return SimilarityFunction.compute_matrix(self, texts)


class JaroWinklerSimilarity(SimilarityFunction):
    """Jaro-Winkler 相似度"""

    def compute(self, text1: str, text2: str) -> float:
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        text1, text2 = text1.lower(), text2.lower()
        len1, len2 = len(text1), len(text2)

        if len1 == 0 and len2 == 0:
            return 1.0

        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        # 查找匹配
        matches = 0
        hash1 = [0] * len1
        hash2 = [0] * len2

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if hash2[j] or text1[i] != text2[j]:
                    continue
                hash1[i] = 1
                hash2[j] = 1
                matches += 1
                break

        if matches == 0:
            return 0.0

        # 计算转置
        transpositions = 0
        point = 0
        for i in range(len1):
            if not hash1[i]:
                continue
            while not hash2[point]:
                point += 1
            if text1[i] != text2[point]:
                transpositions += 1
            point += 1

        jaro = (matches / len1 + matches / len2 +
                (matches - transpositions / 2) / matches) / 3

        # Winkler 前缀修正
        prefix = 0
        for i in range(min(4, len1, len2)):
            if text1[i] == text2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)


class NgramSimilarity(SimilarityFunction):
    """N-gram 相似度"""

    def __init__(self, n: int = 2):
        self.n = n

    def _get_ngrams(self, text: str) -> set:
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngrams.add(text[i:i+self.n])
        return ngrams

    def compute(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        ngrams1 = self._get_ngrams(text1)
        ngrams2 = self._get_ngrams(text2)
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0


class BertSemanticSimilarity(SimilarityFunction):
    """BERT 语义相似度 (使用 sentence-transformers)"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, falling back to Jaccard")
                return False
        return True

    def compute(self, text1: str, text2: str) -> float:
        if not self._load_model():
            return JaccardSimilarity().compute(text1, text2)

        embeddings = self.model.encode([text1, text2])
        # 余弦相似度
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(cos_sim)

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        if not self._load_model():
            return SimilarityFunction.compute_matrix(self, texts)

        embeddings = self.model.encode(texts)
        # 批量计算余弦相似度
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return normalized @ normalized.T


class BleuSimilarity(SimilarityFunction):
    """BLEU 相似度 (词级, 基于 n-gram precision)"""

    def compute(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if not words1 or not words2:
            return 0.0

        # 计算 unigram precision
        matches = 0
        counted = set()
        for w in words1:
            if w in words2 and w not in counted:
                matches += 1
                counted.add(w)

        precision = matches / len(words1) if words1 else 0.0
        # 简化: 直接返回 precision 作为相似度
        return precision


class RougeSimilarity(SimilarityFunction):
    """ROUGE-L 相似度 (最长公共子序列)"""

    def compute(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # LCS 长度
        m, n = len(words1), len(words2)
        if m == 0 or n == 0:
            return 0.0

        # 动态规划 LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]
        # ROUGE-L: LCS / max(len1, len2)
        return lcs_len / max(m, n)


# 相似度函数工厂
SIMILARITY_FUNCTIONS: Dict[SimilarityType, type] = {
    SimilarityType.JACCARD: JaccardSimilarity,
    SimilarityType.LEVENSHTEIN: LevenshteinSimilarity,
    SimilarityType.COSINE: CosineSimilarity,
    SimilarityType.JARO_WINKLER: JaroWinklerSimilarity,
    SimilarityType.NGRAM: NgramSimilarity,
    SimilarityType.BERTOUCH: BertSemanticSimilarity,
    SimilarityType.BLEU: BleuSimilarity,
    SimilarityType.ROUGE: RougeSimilarity,
}


def get_similarity_function(
    similarity_type: SimilarityType,
    **kwargs
) -> SimilarityFunction:
    """获取相似度函数实例"""
    func_class = SIMILARITY_FUNCTIONS.get(similarity_type)
    if func_class is None:
        logger.warning(f"Unknown similarity type: {similarity_type}, using JACCARD")
        func_class = JaccardSimilarity

    # 传递额外参数
    if similarity_type == SimilarityType.NGRAM:
        return func_class(n=kwargs.get('ngram_n', 2))
    elif similarity_type == SimilarityType.BERTOUCH:
        return func_class(model_name=kwargs.get('bert_model', 'sentence-transformers/all-MiniLM-L6-v2'))
    else:
        return func_class()


def compute_diversity_with_similarity(
    data: List[Dict],
    config: DiversityConfig = None,
    similarity_func: Optional[SimilarityFunction] = None,
) -> Dict[str, float]:
    """
    统一的 diversity 计算框架

    通过更换不同的相似度函数计算 diversity:
    - Jaccard: 词级集合相似度
    - Levenshtein: 编辑距离相似度
    - Cosine: TF-IDF 余弦相似度
    - JaroWinkler: Jaro-Winkler 相似度
    - Ngram: N-gram 相似度
    - Bertouch: BERT 语义相似度
    - Bleu: BLEU 相似度
    - Rouge: ROUGE-L 相似度

    Args:
        data: 数据样本列表
        config: DiversityConfig 配置
        similarity_func: 自定义相似度函数 (会覆盖 config)

    Returns:
        Dictionary with diversity metrics
    """
    if config is None:
        config = DiversityConfig()

    if not data:
        return {'diversity_score': 0.0, 'avg_similarity': 0.0}

    # 提取文本
    texts = []
    for item in data:
        if config.compute_on == "prompt":
            prompt = item.get('prompt', '')
            if isinstance(prompt, list):
                text = ' '.join(str(p.get('content', '')) for p in prompt)
            else:
                text = str(prompt)
        elif config.compute_on == "answer":
            answer = item.get('extra_info', {}).get('answer', '')
            text = str(answer) if answer else ''
        else:  # both: prompt + answer
            prompt = item.get('prompt', '')
            if isinstance(prompt, list):
                prompt_text = ' '.join(str(p.get('content', '')) for p in prompt)
            else:
                prompt_text = str(prompt)
            answer = item.get('extra_info', {}).get('answer', '')
            answer_text = str(answer) if answer else ''
            text = prompt_text + " " + answer_text

        if text.strip():
            texts.append(text)

    if len(texts) < 2:
        return {'diversity_score': 0.0, 'avg_similarity': 0.0}

    # 采样
    if config.sample_size and len(texts) > config.sample_size:
        import random
        indices = random.sample(range(len(texts)), config.sample_size)
        texts = [texts[i] for i in indices]

    # 获取相似度函数
    if similarity_func is None:
        similarity_func = get_similarity_function(
            config.similarity_type,
            ngram_n=config.ngram_n,
            bert_model=config.bert_model
        )

    # 计算相似度矩阵
    similarity_matrix = similarity_func.compute_matrix(texts)

    # 提取上三角 (排除对角线)
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    if len(upper_tri) == 0:
        return {'diversity_score': 0.0, 'avg_similarity': 0.0}

    avg_similarity = float(np.mean(upper_tri))
    diversity_score = 1 - avg_similarity

    # 键名加上 similarity_type 前缀以区分不同相似度函数
    sim_type = config.similarity_type.value
    prefix = f"{sim_type}_"

    metrics = {
        f'{prefix}diversity_score': diversity_score,
        f'{prefix}avg_similarity': avg_similarity,
        f'{prefix}min_similarity': float(np.min(upper_tri)),
        f'{prefix}max_similarity': float(np.max(upper_tri)),
        f'{prefix}std_similarity': float(np.std(upper_tri)),
        'diversity_similarity_type': sim_type,
    }

    logger.info(f"Computed diversity with {config.similarity_type.value}: score={diversity_score:.3f}")
    return metrics


# ============================================================================
# 1. TEXT SIMILARITY METRICS
# ============================================================================

def compute_text_similarity_matrix(texts: List[str], method: str = 'jaccard') -> np.ndarray:
    """
    Compute pairwise text similarity matrix

    Args:
        texts: List of text strings
        method: 'jaccard' (word-level), 'cosine' (requires sklearn), 'levenshtein'

    Returns:
        Similarity matrix (n x n)
    """
    n = len(texts)
    similarity_matrix = np.zeros((n, n))

    if method == 'jaccard':
        # Word-level Jaccard similarity
        for i in range(n):
            for j in range(i, n):
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    elif method == 'levenshtein':
        # Levenshtein distance (normalized)
        for i in range(n):
            for j in range(i, n):
                dist = levenshtein_distance(texts[i], texts[j])
                max_len = max(len(texts[i]), len(texts[j]))
                similarity = 1 - (dist / max_len) if max_len > 0 else 1
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    elif method == 'cosine':
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        except ImportError:
            logger.warning("sklearn not available, falling back to jaccard")
            return compute_text_similarity_matrix(texts, method='jaccard')

    return similarity_matrix


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_dataset_diversity(
    data: List[Dict],
    sample_size: int = 100,
    similarity_type: SimilarityType = SimilarityType.JACCARD,
    compute_on: str = "prompt"
) -> Dict[str, float]:
    """
    Compute dataset diversity metrics based on text similarity (统一框架)

    支持多种相似度函数:
    - JACCARD: 词级 Jaccard 相似度
    - LEVENSHTEIN: 编辑距离相似度
    - COSINE: TF-IDF 余弦相似度
    - JARO_WINKLER: Jaro-Winkler 相似度
    - NGRAM: N-gram 相似度
    - BERTOUCH: BERT 语义相似度
    - BLEU: BLEU 相似度
    - ROUGE: ROUGE-L 相似度

    Args:
        data: List of data samples
        sample_size: Number of samples to use for computation
        similarity_type: 相似度函数类型
        compute_on: 计算对象 ("prompt", "answer", "both")

    Returns:
        Dictionary with diversity metrics
    """
    config = DiversityConfig(
        similarity_type=similarity_type,
        sample_size=sample_size,
        compute_on=compute_on
    )
    return compute_diversity_with_similarity(data, config=config)


# ============================================================================
# 2. INFORMATION ENTROPY METRICS
# ============================================================================

def compute_text_entropy(text: str) -> float:
    """
    Compute Shannon entropy of text (character-level)

    Args:
        text: Input text string

    Returns:
        Entropy value
    """
    if not text:
        return 0.0

    # Count character frequencies
    char_counts = Counter(text)
    total_chars = len(text)

    # Compute entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)

    return entropy


def compute_word_entropy(text: str) -> float:
    """
    Compute Shannon entropy of text (word-level)

    Args:
        text: Input text string

    Returns:
        Entropy value
    """
    words = text.lower().split()
    if not words:
        return 0.0

    word_counts = Counter(words)
    total_words = len(words)

    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * math.log2(probability)

    return entropy


def compute_dataset_entropy(data: List[Dict]) -> Dict[str, float]:
    """
    Compute entropy metrics for dataset

    Args:
        data: List of data samples

    Returns:
        Dictionary with entropy metrics
    """
    char_entropies = []
    word_entropies = []

    for item in data:
        if 'prompt' in item:
            prompt = item['prompt']
            if isinstance(prompt, list):
                prompt_text = ' '.join(str(p.get('content', '')) for p in prompt)
            else:
                prompt_text = str(prompt)

            char_entropies.append(compute_text_entropy(prompt_text))
            word_entropies.append(compute_word_entropy(prompt_text))

    metrics = {}
    if char_entropies:
        metrics['avg_char_entropy'] = float(np.mean(char_entropies))
        metrics['std_char_entropy'] = float(np.std(char_entropies))
        metrics['min_char_entropy'] = float(np.min(char_entropies))
        metrics['max_char_entropy'] = float(np.max(char_entropies))

    if word_entropies:
        metrics['avg_word_entropy'] = float(np.mean(word_entropies))
        metrics['std_word_entropy'] = float(np.std(word_entropies))
        metrics['min_word_entropy'] = float(np.min(word_entropies))
        metrics['max_word_entropy'] = float(np.max(word_entropies))

    logger.info(f"Computed entropy metrics: {len(metrics)} metrics")
    return metrics


# ============================================================================
# 3. MULTI-REWARD FUNCTION DIFFICULTY ASSESSMENT
# ============================================================================

@dataclass
class RewardFunctionConfig:
    """Configuration for a reward function"""
    name: str
    func: Callable
    weight: float = 1.0
    description: str = ""


class MultiRewardDifficultyAssessor:
    """Assess sample difficulty using multiple reward functions"""

    def __init__(self, reward_functions: List[RewardFunctionConfig]):
        """
        Initialize assessor with multiple reward functions

        Args:
            reward_functions: List of RewardFunctionConfig objects
        """
        self.reward_functions = reward_functions
        logger.info(f"Initialized MultiRewardDifficultyAssessor with {len(reward_functions)} functions")

    def compute_sample_difficulty(
        self,
        solution: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Compute difficulty scores using all reward functions

        Args:
            solution: Model's solution/response
            ground_truth: Ground truth answer

        Returns:
            Dictionary with scores from each reward function
        """
        scores = {}

        for config in self.reward_functions:
            try:
                score = config.func(solution, ground_truth)
                scores[config.name] = float(score)
            except Exception as e:
                logger.warning(f"Failed to compute score with {config.name}: {e}")
                scores[config.name] = None

        return scores

    def compute_ensemble_difficulty(
        self,
        solution: str,
        ground_truth: str,
        aggregation: str = 'weighted_mean'
    ) -> float:
        """
        Compute ensemble difficulty score

        Args:
            solution: Model's solution
            ground_truth: Ground truth answer
            aggregation: 'weighted_mean', 'mean', 'min', 'max'

        Returns:
            Ensemble difficulty score
        """
        scores = self.compute_sample_difficulty(solution, ground_truth)
        valid_scores = [s for s in scores.values() if s is not None]

        if not valid_scores:
            return 0.5  # Default neutral difficulty

        if aggregation == 'weighted_mean':
            total_weight = sum(c.weight for c in self.reward_functions if c.name in scores and scores[c.name] is not None)
            weighted_sum = sum(
                scores[c.name] * c.weight
                for c in self.reward_functions
                if c.name in scores and scores[c.name] is not None
            )
            return weighted_sum / total_weight if total_weight > 0 else 0.5

        elif aggregation == 'mean':
            return np.mean(valid_scores)
        elif aggregation == 'min':
            return np.min(valid_scores)
        elif aggregation == 'max':
            return np.max(valid_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def compute_dataset_difficulty(
        self,
        data: List[Dict],
        sample_size: int = 100,
        aggregation: str = 'weighted_mean'
    ) -> Dict[str, float]:
        """
        Compute difficulty distribution for dataset

        Args:
            data: List of data samples
            sample_size: Number of samples to evaluate
            aggregation: Aggregation method for ensemble scores

        Returns:
            Dictionary with difficulty statistics
        """
        # Sample data for efficiency
        sample_indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
        sampled_data = [data[i] for i in sample_indices]

        difficulties = []
        individual_scores = {config.name: [] for config in self.reward_functions}

        for item in sampled_data:
            try:
                if 'reward_model' in item and 'ground_truth' in item['reward_model']:
                    ground_truth = item['reward_model']['ground_truth']
                    solution = item.get('extra_info', {}).get('answer', '')

                    # Compute individual scores
                    scores = self.compute_sample_difficulty(solution, ground_truth)
                    for name, score in scores.items():
                        if score is not None:
                            individual_scores[name].append(score)

                    # Compute ensemble difficulty (1 - score, so higher score = easier = lower difficulty)
                    ensemble_score = self.compute_ensemble_difficulty(solution, ground_truth, aggregation)
                    difficulty = 1 - ensemble_score  # Convert to difficulty
                    difficulties.append(difficulty)

            except Exception as e:
                logger.warning(f"Failed to compute difficulty: {e}")

        metrics = {}

        # Ensemble difficulty statistics
        if difficulties:
            metrics['avg_difficulty'] = float(np.mean(difficulties))
            metrics['std_difficulty'] = float(np.std(difficulties))
            metrics['min_difficulty'] = float(np.min(difficulties))
            metrics['max_difficulty'] = float(np.max(difficulties))

            # Difficulty distribution
            easy_threshold = np.percentile(difficulties, 33)
            hard_threshold = np.percentile(difficulties, 67)
            metrics['easy_ratio'] = float(np.mean(np.array(difficulties) <= easy_threshold))
            metrics['medium_ratio'] = float(np.mean((np.array(difficulties) > easy_threshold) & (np.array(difficulties) <= hard_threshold)))
            metrics['hard_ratio'] = float(np.mean(np.array(difficulties) > hard_threshold))

        # Individual reward function statistics
        for name, scores in individual_scores.items():
            if scores:
                metrics[f'{name}_avg_score'] = float(np.mean(scores))
                metrics[f'{name}_std_score'] = float(np.std(scores))
                metrics[f'{name}_agreement'] = float(np.mean([1 if s >= 0.5 else 0 for s in scores]))

        logger.info(f"Computed difficulty for {len(difficulties)} samples using {len(self.reward_functions)} reward functions")
        return metrics


# ============================================================================
# 4. INTEGRATED ADVANCED METRICS
# ============================================================================

def compute_advanced_data_metrics(
    data: List[Dict],
    reward_functions: Optional[List[RewardFunctionConfig]] = None,
    include_similarity: bool = True,
    include_entropy: bool = True,
    include_difficulty: bool = True,
    sample_size: int = 100,
    diversity_similarity_type: SimilarityType = SimilarityType.JACCARD,
    diversity_compute_on: str = "prompt"
) -> Dict[str, float]:
    """
    Compute all advanced data metrics

    Args:
        data: List of data samples
        reward_functions: List of reward function configs for difficulty assessment
        include_similarity: Whether to compute text similarity metrics
        include_entropy: Whether to compute entropy metrics
        include_difficulty: Whether to compute difficulty metrics
        sample_size: Sample size for computation
        diversity_similarity_type: 相似度函数类型 (JACCARD, LEVENSHTEIN, COSINE, etc.)
        diversity_compute_on: 计算对象 ("prompt", "answer", "both")

    Returns:
        Dictionary with all advanced metrics
    """
    all_metrics = {}

    # Text similarity and diversity
    if include_similarity:
        try:
            diversity_metrics = compute_dataset_diversity(
                data,
                sample_size=sample_size,
                similarity_type=diversity_similarity_type,
                compute_on=diversity_compute_on
            )
            all_metrics.update(diversity_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute diversity metrics: {e}")

    # Information entropy
    if include_entropy:
        try:
            entropy_metrics = compute_dataset_entropy(data)
            all_metrics.update(entropy_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute entropy metrics: {e}")

    # Multi-reward difficulty assessment
    if include_difficulty and reward_functions:
        try:
            assessor = MultiRewardDifficultyAssessor(reward_functions)
            difficulty_metrics = assessor.compute_dataset_difficulty(data, sample_size)
            all_metrics.update(difficulty_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute difficulty metrics: {e}")

    return all_metrics


# ============================================================================
# 5. EXAMPLE REWARD FUNCTIONS
# ============================================================================

def reward_exact_match(solution: str, ground_truth: str) -> float:
    """Exact match reward"""
    return 1.0 if solution.strip() == ground_truth.strip() else 0.0


def reward_partial_match(solution: str, ground_truth: str) -> float:
    """Partial match reward (word overlap)"""
    solution_words = set(solution.lower().split())
    truth_words = set(ground_truth.lower().split())
    if not truth_words:
        return 0.0
    overlap = len(solution_words & truth_words)
    return overlap / len(truth_words)


def reward_contains(solution: str, ground_truth: str) -> float:
    """Reward if solution contains ground truth"""
    return 1.0 if ground_truth.lower() in solution.lower() else 0.0


def reward_similarity(solution: str, ground_truth: str) -> float:
    """Reward based on string similarity"""
    dist = levenshtein_distance(solution, ground_truth)
    max_len = max(len(solution), len(ground_truth))
    return 1 - (dist / max_len) if max_len > 0 else 1.0


# ============================================================================
# 3. PERPLEXITY (PPL) METRICS
# ============================================================================

def compute_ppl_for_text(text: str, model, tokenizer, device: str = 'cuda') -> float:
    """
    Compute perplexity for a single text using the given model.

    PPL = exp(-1/N * sum(log_p(x_i)))

    Args:
        text: Input text
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        device: Device to run computation

    Returns:
        Perplexity value
    """
    import torch
    from torch.nn import functional as F

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    # Compute loss
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # Shift for causal LM: predict next token
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

    ppl = torch.exp(loss).item()
    return ppl


def compute_ppl_metrics(
    data: List[Dict],
    model_name: str = "gpt2",
    device: str = "cuda",
    max_samples: int = None
) -> Dict[str, float]:
    """
    Compute perplexity metrics for dataset answers.

    PPL uses a pre-trained model to compute the model's uncertainty over the text.
    - High PPL: model is uncertain (could be noisy/random text)
    - Low PPL: model is confident (could be generic/boilerplate text)

    Args:
        data: List of data samples
        model_name: HuggingFace model name or path
        device: Device to run on
        max_samples: Maximum samples to compute (for speed)

    Returns:
        Dictionary with PPL metrics
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("transformers/torch not available, skipping PPL computation")
        return {'avg_ppl': 0.0, 'std_ppl': 0.0, 'min_ppl': 0.0, 'max_ppl': 0.0}

    # Limit samples for speed
    if max_samples and len(data) > max_samples:
        import random
        data = random.sample(data, max_samples)

    # Extract answers
    answers = []
    for item in data:
        answer = None
        # Try different fields
        if 'extra_info' in item and 'answer' in item['extra_info']:
            answer = str(item['extra_info']['answer'])
        elif 'answer' in item:
            answer = str(item['answer'])

        if answer and answer.strip():
            answers.append(answer)

    if not answers:
        logger.warning("No answers found in data for PPL computation")
        return {'avg_ppl': 0.0, 'std_ppl': 0.0, 'min_ppl': 0.0, 'max_ppl': 0.0}

    # Load model
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return {'avg_ppl': 0.0, 'std_ppl': 0.0, 'min_ppl': 0.0, 'max_ppl': 0.0}

    # Compute PPL for each answer
    ppls = []
    for i, answer in enumerate(answers):
        if i % 100 == 0:
            logger.info(f"Computing PPL for sample {i}/{len(answers)}")
        try:
            ppl = compute_ppl_for_text(answer, model, tokenizer, device)
            if not np.isnan(ppl) and not np.isinf(ppl):
                ppls.append(ppl)
        except Exception as e:
            logger.warning(f"Failed to compute PPL for sample {i}: {e}")
            continue

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not ppls:
        logger.warning("No valid PPL values computed")
        return {'avg_ppl': 0.0, 'std_ppl': 0.0, 'min_ppl': 0.0, 'max_ppl': 0.0}

    metrics = {
        'avg_ppl': float(np.mean(ppls)),
        'std_ppl': float(np.std(ppls)),
        'min_ppl': float(np.min(ppls)),
        'max_ppl': float(np.max(ppls)),
    }

    logger.info(f"Computed PPL metrics: avg={metrics['avg_ppl']:.2f}, std={metrics['std_ppl']:.2f}")
    return metrics


# ============================================================================
# 4. IFD (INSTRUCTION FOLLOWING DIFFICULTY) METRICS
# ============================================================================

def compute_ifd_for_sample(prompt: str, answer: str, model, tokenizer, device: str = 'cuda') -> float:
    """
    Compute IFD (Instruction Following Difficulty) for a single sample.

    IFD = Loss(with_prompt) / Loss(without_prompt)

    Args:
        prompt: Input prompt
        answer: Expected answer
        model: Pre-trained language model
        tokenizer: Tokenizer
        device: Device

    Returns:
        IFD value
    """
    import torch
    from torch.nn import functional as F

    # === With Prompt ===
    # Encode: prompt + answer
    prompt_text = prompt if prompt else ""
    full_text = prompt_text + answer
    encodings = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # Get loss for answer tokens only (shift by prompt length)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        # For causal LM, loss covers all tokens, we approximate by full loss
        loss_with_prompt = outputs.loss.item() if hasattr(outputs, 'loss') else outputs[0].item()

    # === Without Prompt (answer only) ===
    answer_encodings = tokenizer(answer, return_tensors='pt', truncation=True, max_length=2048)
    answer_input_ids = answer_encodings.input_ids.to(device)

    with torch.no_grad():
        answer_outputs = model(answer_input_ids, labels=answer_input_ids)
        loss_without_prompt = answer_outputs.loss.item() if hasattr(answer_outputs, 'loss') else answer_outputs[0].item()

    # Compute IFD
    if loss_without_prompt > 0:
        ifd = loss_with_prompt / loss_without_prompt
    else:
        ifd = 0.0

    return ifd


def compute_ifd_metrics(
    data: List[Dict],
    model_name: str = "gpt2",
    device: str = "cuda",
    max_samples: int = None
) -> Dict[str, float]:
    """
    Compute IFD (Instruction Following Difficulty) metrics for dataset.

    IFD = Loss(with_prompt) / Loss(without_prompt)

    - High IFD: Answer is hard to predict from prompt alone (more learning value)
    - Low IFD: Answer is easy to predict (might be generic)

    Args:
        data: List of data samples
        model_name: HuggingFace model name or path
        device: Device to run on
        max_samples: Maximum samples to compute

    Returns:
        Dictionary with IFD metrics
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("transformers/torch not available, skipping IFD computation")
        return {'avg_ifd': 0.0, 'std_ifd': 0.0, 'min_ifd': 0.0, 'max_ifd': 0.0}

    # Limit samples for speed
    if max_samples and len(data) > max_samples:
        import random
        data = random.sample(data, max_samples)

    # Extract prompt-answer pairs
    pairs = []
    for item in data:
        prompt = None
        answer = None

        # Get prompt
        if 'prompt' in item:
            if isinstance(item['prompt'], list):
                prompt = ' '.join(str(p.get('content', '')) for p in item['prompt'])
            else:
                prompt = str(item['prompt'])

        # Get answer (try different fields)
        if 'extra_info' in item and 'answer' in item['extra_info']:
            answer = str(item['extra_info']['answer'])
        elif 'answer' in item:
            answer = str(item['answer'])
        elif 'reward_model' in item and 'ground_truth' in item['reward_model']:
            answer = str(item['reward_model']['ground_truth'])

        if prompt and answer and prompt.strip() and answer.strip():
            pairs.append((prompt, answer))

    if not pairs:
        logger.warning("No valid prompt-answer pairs found for IFD computation")
        return {'avg_ifd': 0.0, 'std_ifd': 0.0, 'min_ifd': 0.0, 'max_ifd': 0.0}

    # Load model
    logger.info(f"Loading model for IFD: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return {'avg_ifd': 0.0, 'std_ifd': 0.0, 'min_ifd': 0.0, 'max_ifd': 0.0}

    # Compute IFD for each pair
    ifds = []
    for i, (prompt, answer) in enumerate(pairs):
        if i % 100 == 0:
            logger.info(f"Computing IFD for sample {i}/{len(pairs)}")
        try:
            ifd = compute_ifd_for_sample(prompt, answer, model, tokenizer, device)
            if not np.isnan(ifd) and not np.isinf(ifd) and ifd > 0:
                ifds.append(ifd)
        except Exception as e:
            logger.warning(f"Failed to compute IFD for sample {i}: {e}")
            continue

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not ifds:
        logger.warning("No valid IFD values computed")
        return {'avg_ifd': 0.0, 'std_ifd': 0.0, 'min_ifd': 0.0, 'max_ifd': 0.0}

    metrics = {
        'avg_ifd': float(np.mean(ifds)),
        'std_ifd': float(np.std(ifds)),
        'min_ifd': float(np.min(ifds)),
        'max_ifd': float(np.max(ifds)),
    }

    logger.info(f"Computed IFD metrics: avg={metrics['avg_ifd']:.2f}, std={metrics['std_ifd']:.2f}")
    return metrics
