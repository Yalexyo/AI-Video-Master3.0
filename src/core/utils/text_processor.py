import os
import json
import re
import jieba
import jieba.posseg as pseg
import logging
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextProcessor:
    """文本处理工具类，封装文本分析与处理相关功能"""
    
    def __init__(self, model_path=None):
        """
        初始化文本处理器
        
        Args:
            model_path: 语义模型路径，默认为None则使用预训练模型
        """
        logger.info("初始化文本处理器")
        
        # 加载语义模型
        self.model_path = model_path or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        try:
            self.encoder = SentenceTransformer(self.model_path)
            logger.info(f"语义编码模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"语义编码模型加载失败: {str(e)}")
            self.encoder = None
    
    def segment_text(self, text):
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            list: 分词结果
        """
        try:
            words = list(jieba.cut(text))
            return words
        except Exception as e:
            logger.error(f"文本分词失败: {str(e)}")
            return []
    
    def get_pos_tags(self, text):
        """
        获取文本词性标注
        
        Args:
            text: 输入文本
            
        Returns:
            list: 词性标注结果，每个元素为(词, 词性)元组
        """
        try:
            words_pos = list(pseg.cut(text))
            return [(word, flag) for word, flag in words_pos]
        except Exception as e:
            logger.error(f"词性标注失败: {str(e)}")
            return []
    
    def extract_keywords(self, text, top_n=10, use_tfidf=True):
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
            top_n: 返回的关键词数量
            use_tfidf: 是否使用TF-IDF算法，否则使用词频
            
        Returns:
            list: 关键词列表，每个元素为(关键词, 权重)元组
        """
        try:
            if use_tfidf:
                # 使用jieba的TF-IDF提取
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(text, topK=top_n, withWeight=True)
                return keywords
            else:
                # 使用词频提取
                words = self.segment_text(text)
                word_counter = Counter(words)
                
                # 过滤掉停用词和标点符号
                filtered_words = [(word, count) for word, count in word_counter.most_common(top_n*2)
                                 if len(word) > 1 and not self._is_stopword(word)]
                
                return filtered_words[:top_n]
                
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return []
    
    def _is_stopword(self, word):
        """
        判断是否为停用词或标点符号
        
        Args:
            word: 输入词
            
        Returns:
            bool: 是否为停用词
        """
        # 简化版，实际应该使用停用词表
        pattern = r'^[^\w\s]$'
        return bool(re.match(pattern, word))
    
    def extract_noun_keywords(self, text, top_n=10):
        """
        从文本中提取名词关键词
        
        Args:
            text: 输入文本
            top_n: 返回的关键词数量
            
        Returns:
            list: 名词关键词列表
        """
        try:
            # 获取词性标注
            words_pos = self.get_pos_tags(text)
            
            # 筛选名词 (n开头的词性)
            nouns = [word for word, flag in words_pos if flag.startswith('n')]
            
            # 统计词频
            counter = Counter(nouns)
            
            # 返回出现频率最高的top_n个名词
            return [word for word, _ in counter.most_common(top_n)]
            
        except Exception as e:
            logger.error(f"名词关键词提取失败: {str(e)}")
            return []
    
    def split_sentences(self, text):
        """
        对文本进行分句
        
        Args:
            text: 输入文本
            
        Returns:
            list: 句子列表
        """
        try:
            # 使用正则表达式匹配句子边界
            pattern = r'[^!?。！？\.\n]+[!?。！？\.\n]'
            sentences = re.findall(pattern, text)
            
            # 处理可能缺失的最后一句
            if text and not text.endswith(('!', '?', '。', '！', '？', '.')):
                last_part = text[sum(len(s) for s in sentences):]
                if last_part.strip():
                    sentences.append(last_part.strip())
            
            return [s.strip() for s in sentences if s.strip()]
            
        except Exception as e:
            logger.error(f"文本分句失败: {str(e)}")
            return [text] if text else []
    
    def compute_sentence_embeddings(self, sentences):
        """
        计算句子的语义嵌入向量
        
        Args:
            sentences: 句子列表
            
        Returns:
            numpy.ndarray: 句子嵌入矩阵，每行为一个句子的嵌入向量
        """
        if not self.encoder:
            logger.error("语义编码模型未加载，无法计算句子嵌入")
            return None
            
        try:
            embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"计算句子嵌入失败: {str(e)}")
            return None
    
    def compute_semantic_similarity(self, text1, text2):
        """
        计算两段文本的语义相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            float: 相似度分数，范围[0,1]
        """
        if not self.encoder:
            logger.error("语义编码模型未加载，无法计算语义相似度")
            return 0.0
            
        try:
            # 计算文本嵌入
            embedding1 = self.encoder.encode(text1, convert_to_numpy=True)
            embedding2 = self.encoder.encode(text2, convert_to_numpy=True)
            
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算语义相似度失败: {str(e)}")
            return 0.0
    
    def intent_keyword_matching(self, text, intent_keywords):
        """
        基于关键词匹配计算文本与意图的匹配分数
        
        Args:
            text: 输入文本
            intent_keywords: 意图关键词列表
            
        Returns:
            float: 匹配分数，范围[0,1]
        """
        try:
            # 分词
            words = self.segment_text(text)
            
            # 计算关键词匹配数量
            matched_keywords = [keyword for keyword in intent_keywords if keyword in words]
            
            # 计算匹配分数
            if not intent_keywords:
                return 0.0
                
            match_score = len(matched_keywords) / len(intent_keywords)
            
            return match_score
            
        except Exception as e:
            logger.error(f"意图关键词匹配失败: {str(e)}")
            return 0.0
    
    def compute_text_features(self, text):
        """
        计算文本的特征
        
        Args:
            text: 输入文本
            
        Returns:
            dict: 文本特征字典
        """
        try:
            # 分词
            words = self.segment_text(text)
            
            # 句子数量
            sentences = self.split_sentences(text)
            
            # 词性标注
            words_pos = self.get_pos_tags(text)
            
            # 统计不同词性的数量
            pos_counts = Counter([pos for _, pos in words_pos])
            
            # 计算特征
            features = {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "noun_ratio": pos_counts.get('n', 0) / len(words_pos) if words_pos else 0,
                "verb_ratio": pos_counts.get('v', 0) / len(words_pos) if words_pos else 0,
                "adj_ratio": pos_counts.get('a', 0) / len(words_pos) if words_pos else 0,
                "unique_word_ratio": len(set(words)) / len(words) if words else 0
            }
            
            # 添加语义嵌入
            if self.encoder:
                try:
                    embedding = self.encoder.encode(text, convert_to_numpy=True)
                    features["embedding"] = embedding.tolist()
                except:
                    pass
            
            return features
            
        except Exception as e:
            logger.error(f"计算文本特征失败: {str(e)}")
            return {}
    
    def save_text_analysis(self, analysis, output_path):
        """
        保存文本分析结果到文件
        
        Args:
            analysis: 文本分析结果
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"文本分析结果已保存到 {output_path}")
        except Exception as e:
            logger.error(f"保存文本分析结果失败: {str(e)}")
    
    def batch_compute_similarities(self, query_text, candidate_texts):
        """
        批量计算查询文本与候选文本的相似度
        
        Args:
            query_text: 查询文本
            candidate_texts: 候选文本列表
            
        Returns:
            list: 相似度分数列表
        """
        if not self.encoder:
            logger.error("语义编码模型未加载，无法计算相似度")
            return [0.0] * len(candidate_texts)
            
        try:
            # 计算查询文本嵌入
            query_embedding = self.encoder.encode(query_text, convert_to_numpy=True)
            
            # 计算候选文本嵌入
            candidate_embeddings = self.encoder.encode(candidate_texts, convert_to_numpy=True)
            
            # 计算余弦相似度
            similarities = []
            for candidate_embedding in candidate_embeddings:
                similarity = np.dot(query_embedding, candidate_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding))
                similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            logger.error(f"批量计算相似度失败: {str(e)}")
            return [0.0] * len(candidate_texts) 