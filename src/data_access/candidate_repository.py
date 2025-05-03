import os
import json
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class CandidateRepository:
    """候选片段存储库，管理和检索候选视频片段"""
    
    def __init__(self, storage_dir="data/processed/candidates"):
        """
        初始化候选片段存储库
        
        Args:
            storage_dir: 存储目录
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.index_path = os.path.join(storage_dir, "index.json")
        self.segments_path = os.path.join(storage_dir, "segments.json")
        
        # 候选片段索引，结构为：
        # {
        #   segment_id: {metadata}
        # }
        self.index = {}
        
        # 候选片段数据，结构为：
        # {
        #   segment_id: {完整数据}
        # }
        self.segments = {}
        
        # 文本特征索引，用于快速检索
        self.text_features = {}
        
        # 视觉标签索引
        self.visual_tags_index = defaultdict(list)
        
        # 加载已有数据（如果存在）
        self._load_data()
        
        logger.info(f"初始化候选片段存储库，存储目录: {storage_dir}")
    
    def _load_data(self):
        """加载已有的索引和片段数据"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                logger.info(f"已加载索引数据，共 {len(self.index)} 条记录")
            
            if os.path.exists(self.segments_path):
                with open(self.segments_path, 'r', encoding='utf-8') as f:
                    self.segments = json.load(f)
                logger.info(f"已加载片段数据，共 {len(self.segments)} 条记录")
                
                # 重建索引
                self._rebuild_indexes()
                
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            # 初始化为空
            self.index = {}
            self.segments = {}
    
    def _rebuild_indexes(self):
        """重建内存索引"""
        # 重建视觉标签索引
        self.visual_tags_index = defaultdict(list)
        
        for segment_id, segment in self.segments.items():
            # 建立视觉标签索引
            for tag in segment.get("visual_tags", []):
                tag_type = tag.get("type")
                tag_name = tag.get("name")
                
                if tag_type and tag_name:
                    tag_key = f"{tag_type}:{tag_name}"
                    self.visual_tags_index[tag_key].append(segment_id)
            
            # 存储文本特征
            if "text_features" in segment:
                self.text_features[segment_id] = segment["text_features"]
        
        logger.info(f"索引重建完成，视觉标签索引包含 {len(self.visual_tags_index)} 个标签")
    
    def save_data(self):
        """保存索引和片段数据到文件"""
        try:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            
            with open(self.segments_path, 'w', encoding='utf-8') as f:
                json.dump(self.segments, f, ensure_ascii=False, indent=2)
                
            logger.info(f"数据保存成功，共 {len(self.segments)} 条片段记录")
            return True
            
        except Exception as e:
            logger.error(f"数据保存失败: {str(e)}")
            return False
    
    def add_segment(self, segment):
        """
        添加一个候选片段
        
        Args:
            segment: 片段数据，必须包含唯一的segment_id
            
        Returns:
            bool: 是否成功
        """
        segment_id = segment.get("segment_id")
        
        if not segment_id:
            logger.error("片段数据缺少segment_id字段")
            return False
            
        try:
            # 添加到索引
            self.index[segment_id] = {
                "source_video": segment.get("source_video", ""),
                "start_time": segment.get("start_time", 0),
                "end_time": segment.get("end_time", 0),
                "duration": segment.get("duration", 0),
                "text_content": segment.get("text_content", ""),
                "visual_tag_summary": self._summarize_visual_tags(segment.get("visual_tags", []))
            }
            
            # 添加到片段数据
            self.segments[segment_id] = segment
            
            # 更新索引
            for tag in segment.get("visual_tags", []):
                tag_type = tag.get("type")
                tag_name = tag.get("name")
                
                if tag_type and tag_name:
                    tag_key = f"{tag_type}:{tag_name}"
                    self.visual_tags_index[tag_key].append(segment_id)
            
            # 存储文本特征
            if "text_features" in segment:
                self.text_features[segment_id] = segment["text_features"]
            
            logger.info(f"添加片段成功: {segment_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加片段失败: {str(e)}")
            return False
    
    def _summarize_visual_tags(self, visual_tags):
        """生成视觉标签摘要"""
        summary = {}
        
        for tag in visual_tags:
            tag_type = tag.get("type")
            tag_name = tag.get("name")
            
            if tag_type and tag_name:
                if tag_type not in summary:
                    summary[tag_type] = []
                
                if tag_name not in summary[tag_type]:
                    summary[tag_type].append(tag_name)
        
        return summary
    
    def get_segment(self, segment_id):
        """
        获取指定ID的片段
        
        Args:
            segment_id: 片段ID
            
        Returns:
            dict: 片段数据，不存在则返回None
        """
        return self.segments.get(segment_id)
    
    def get_all_segments(self):
        """
        获取所有片段
        
        Returns:
            list: 所有片段数据的列表
        """
        return list(self.segments.values())
    
    def search_by_visual_tags(self, tags, match_all=False):
        """
        根据视觉标签搜索片段
        
        Args:
            tags: 标签列表，每个标签为"类型:名称"格式
            match_all: 是否要求匹配所有标签
            
        Returns:
            list: 匹配的片段ID列表
        """
        if not tags:
            return []
            
        # 获取每个标签匹配的片段ID
        matched_sets = []
        
        for tag in tags:
            if tag in self.visual_tags_index:
                matched_sets.append(set(self.visual_tags_index[tag]))
            else:
                # 如果要求匹配所有标签，但有一个标签没有匹配项，则返回空
                if match_all:
                    return []
                matched_sets.append(set())
        
        if not matched_sets:
            return []
            
        # 根据匹配模式计算结果
        if match_all:
            # 求交集
            result = set.intersection(*matched_sets)
        else:
            # 求并集
            result = set.union(*matched_sets)
        
        return list(result)
    
    def search_by_text_content(self, text_processor, query_text, top_n=10):
        """
        根据文本内容搜索片段
        
        Args:
            text_processor: TextProcessor实例，用于计算文本相似度
            query_text: 查询文本
            top_n: 返回结果数量
            
        Returns:
            list: 匹配的片段ID列表，按相似度降序排序
        """
        if not text_processor or not query_text or not self.segments:
            return []
            
        try:
            # 获取所有片段的文本内容
            segment_ids = []
            texts = []
            
            for segment_id, segment in self.segments.items():
                text_content = segment.get("text_content", "")
                if text_content:
                    segment_ids.append(segment_id)
                    texts.append(text_content)
            
            if not texts:
                return []
                
            # 计算相似度
            similarities = text_processor.batch_compute_similarities(query_text, texts)
            
            # 组合ID和相似度
            id_sim_pairs = list(zip(segment_ids, similarities))
            
            # 按相似度降序排序
            id_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_n个结果
            return [pair[0] for pair in id_sim_pairs[:top_n]]
            
        except Exception as e:
            logger.error(f"文本搜索失败: {str(e)}")
            return []
    
    def search_by_keywords(self, keywords):
        """
        根据关键词搜索片段
        
        Args:
            keywords: 关键词列表
            
        Returns:
            list: 匹配的片段ID列表，按匹配关键词数量降序排序
        """
        if not keywords or not self.segments:
            return []
            
        # 遍历所有片段，计算关键词匹配数
        matches = []
        
        for segment_id, segment in self.segments.items():
            text_content = segment.get("text_content", "")
            if not text_content:
                continue
                
            # 简单的关键词匹配，实际应用中可能需要更复杂的算法
            match_count = sum(1 for keyword in keywords if keyword in text_content)
            
            if match_count > 0:
                matches.append((segment_id, match_count))
        
        # 按匹配关键词数量降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # 返回匹配的片段ID
        return [match[0] for match in matches]
    
    def compute_segment_match_score(self, segment_id, intent_data, weights=None):
        """
        计算片段与意图的匹配分数
        
        Args:
            segment_id: 片段ID
            intent_data: 意图数据，包含关键词和期望视觉元素
            weights: 权重字典，默认为None
            
        Returns:
            float: 匹配分数
        """
        if segment_id not in self.segments:
            return 0.0
            
        # 默认权重
        if weights is None:
            weights = {
                "visual": 0.4,
                "keyword": 0.4,
                "semantic": 0.2
            }
            
        segment = self.segments[segment_id]
        
        # 1. 计算视觉分数
        visual_score = 0.0
        expected_visual_elements = intent_data.get("expected_visual_elements", [])
        
        if expected_visual_elements:
            segment_visual_tags = [f"{tag.get('type')}:{tag.get('name')}" for tag in segment.get("visual_tags", [])]
            
            # 计算重叠度
            overlap_count = sum(1 for elem in expected_visual_elements if elem in segment_visual_tags)
            visual_score = overlap_count / len(expected_visual_elements) if expected_visual_elements else 0.0
        
        # 2. 计算关键词分数
        keyword_score = 0.0
        keywords = intent_data.get("keywords", [])
        
        if keywords:
            text_content = segment.get("text_content", "")
            
            # 计算包含的关键词数量
            included_count = sum(1 for keyword in keywords if keyword in text_content)
            keyword_score = included_count / len(keywords) if keywords else 0.0
        
        # 3. 语义分数（假设已预先计算并存储在segment中）
        semantic_score = segment.get("semantic_score", 0.0)
        
        # 计算综合分数
        total_score = (
            weights["visual"] * visual_score +
            weights["keyword"] * keyword_score +
            weights["semantic"] * semantic_score
        )
        
        return total_score
    
    def segment_statistics(self):
        """
        计算片段库的统计信息
        
        Returns:
            dict: 统计信息
        """
        if not self.segments:
            return {
                "total_segments": 0,
                "total_duration": 0,
                "source_videos": 0,
                "avg_segment_duration": 0
            }
            
        # 计算总时长
        total_duration = sum(segment.get("duration", 0) for segment in self.segments.values())
        
        # 计算源视频数量
        source_videos = set(segment.get("source_video", "") for segment in self.segments.values())
        
        # 计算平均片段时长
        avg_duration = total_duration / len(self.segments) if self.segments else 0
        
        # 统计视觉标签
        visual_tag_counts = defaultdict(int)
        
        for segment in self.segments.values():
            for tag in segment.get("visual_tags", []):
                tag_type = tag.get("type")
                tag_name = tag.get("name")
                
                if tag_type and tag_name:
                    visual_tag_counts[f"{tag_type}:{tag_name}"] += 1
        
        # 获取前20个最常见的视觉标签
        top_tags = sorted(
            [(tag, count) for tag, count in visual_tag_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            "total_segments": len(self.segments),
            "total_duration": total_duration,
            "source_videos": len(source_videos),
            "avg_segment_duration": avg_duration,
            "top_visual_tags": dict(top_tags)
        } 