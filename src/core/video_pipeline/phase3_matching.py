import os
import json
import logging
import random
from collections import defaultdict

from src.core.utils.text_processor import TextProcessor
from src.data_access.candidate_repository import CandidateRepository

logger = logging.getLogger(__name__)

class MatchingPipeline:
    """第三阶段处理流程 - 高效匹配，实现抽象脚本与候选片段的匹配"""
    
    def __init__(self, abstract_script_path=None, candidates_dir="data/processed/candidates", output_dir="data/processed"):
        """
        初始化高效匹配流水线
        
        Args:
            abstract_script_path: 抽象脚本路径，默认为None
            candidates_dir: 候选片段库目录
            output_dir: 输出目录
        """
        self.abstract_script_path = abstract_script_path
        self.candidates_dir = candidates_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        self.matching_dir = os.path.join(output_dir, "matching")
        os.makedirs(self.matching_dir, exist_ok=True)
        
        # 初始化处理器和存储库
        self.text_processor = TextProcessor()
        self.repository = CandidateRepository(storage_dir=candidates_dir)
        
        # 权重配置
        self.default_weights = {
            "visual": 0.4,
            "keyword": 0.4,
            "semantic": 0.2
        }
        
        # 匹配参数
        self.top_k = 10  # 每个意图段落保留的候选数量
        
        logger.info(f"初始化高效匹配流水线，抽象脚本路径: {abstract_script_path}, 候选片段库目录: {candidates_dir}")
    
    def run(self, abstract_script_path=None, weights=None, diversity_threshold=0.3):
        """
        运行高效匹配流水线
        
        Args:
            abstract_script_path: 抽象脚本路径，默认为None
            weights: 匹配权重配置，默认为None
            diversity_threshold: 多样性阈值，控制同一源视频在同一意图段落的候选数量比例
            
        Returns:
            str: 匹配结果路径
        """
        # 使用传入的抽象脚本路径或初始化时设置的路径
        script_path = abstract_script_path or self.abstract_script_path
        
        if not script_path:
            logger.error("未指定抽象脚本路径")
            return None
        
        # 使用传入的权重或默认权重
        weights = weights or self.default_weights
        
        logger.info("开始运行高效匹配流水线")
        
        # 步骤3.1: 加载抽象脚本和候选片段库
        abstract_script = self._load_abstract_script(script_path)
        
        if not abstract_script:
            logger.error("加载抽象脚本失败")
            return None
        
        # 获取候选片段库统计信息
        repo_stats = self.repository.segment_statistics()
        logger.info(f"候选片段库包含 {repo_stats.get('total_segments', 0)} 个片段")
        
        # 步骤3.2: 定义匹配目标
        matching_targets = self._define_matching_targets(abstract_script)
        
        # 步骤3.3: 候选片段评分与匹配
        matching_results = self._match_segments(matching_targets, weights, diversity_threshold)
        
        # 保存匹配结果
        output_path = os.path.join(self.matching_dir, "matching_results.json")
        self._save_matching_results(matching_results, output_path)
        
        logger.info(f"高效匹配流水线运行完成，结果保存到: {output_path}")
        return output_path
    
    def _load_abstract_script(self, script_path):
        """
        加载抽象脚本
        
        Args:
            script_path: 抽象脚本路径
            
        Returns:
            dict: 抽象脚本对象
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script = json.load(f)
                
            logger.info(f"抽象脚本加载成功: {script_path}")
            return script
        except Exception as e:
            logger.error(f"加载抽象脚本失败: {str(e)}")
            return None
    
    def _define_matching_targets(self, abstract_script):
        """
        根据抽象脚本定义匹配目标
        
        Args:
            abstract_script: 抽象脚本对象
            
        Returns:
            list: 匹配目标列表，每个元素包含意图类型、关键词和期望视觉元素
        """
        matching_targets = []
        
        # 获取意图序列和意图详情
        intent_sequence = abstract_script.get("intent_sequence", [])
        intent_details = abstract_script.get("intent_details", [])
        
        # 将两者匹配起来
        for i, intent_type in enumerate(intent_sequence):
            # 查找对应的意图详情
            detail = None
            for d in intent_details:
                if d.get("intent_type") == intent_type:
                    detail = d
                    break
            
            if not detail:
                logger.warning(f"未找到意图类型 {intent_type} 的详情")
                keywords = []
            else:
                keywords = detail.get("keywords", [])
            
            # 根据意图类型推断期望视觉元素
            expected_visual_elements = self._infer_expected_visual_elements(intent_type)
            
            # 构建匹配目标
            target = {
                "index": i,
                "intent_type": intent_type,
                "keywords": keywords,
                "expected_visual_elements": expected_visual_elements
            }
            
            matching_targets.append(target)
            logger.info(f"定义匹配目标: {intent_type}, 关键词数量: {len(keywords)}, 期望视觉元素数量: {len(expected_visual_elements)}")
        
        return matching_targets
    
    def _infer_expected_visual_elements(self, intent_type):
        """
        根据意图类型推断期望视觉元素
        
        Args:
            intent_type: 意图类型
            
        Returns:
            list: 期望视觉元素列表
        """
        # 根据不同意图类型定义期望的视觉元素
        elements = []
        
        if intent_type == "问题引入":
            elements = [
                "scene:家庭",
                "action:抱婴儿",
                "object:婴儿",
                "object:妈妈"
            ]
        elif intent_type == "产品介绍":
            elements = [
                "product:产品特写",
                "product:产品展示",
                "object:奶瓶",
                "object:包装"
            ]
        elif intent_type == "效果展示":
            elements = [
                "action:喂奶",
                "object:婴儿",
                "scene:家庭",
                "object:奶瓶"
            ]
        elif intent_type == "促销信息":
            elements = [
                "product:产品展示",
                "product:产品特写",
                "object:包装"
            ]
        
        return elements
    
    def _match_segments(self, matching_targets, weights, diversity_threshold):
        """
        为每个意图段落匹配候选片段
        
        Args:
            matching_targets: 匹配目标列表
            weights: 匹配权重配置
            diversity_threshold: 多样性阈值
            
        Returns:
            dict: 匹配结果
        """
        matching_results = {}
        
        for target in matching_targets:
            intent_type = target["intent_type"]
            
            logger.info(f"匹配意图段落: {intent_type}")
            
            # 第一步：根据视觉标签筛选候选片段
            visual_segments = self._filter_by_visual_tags(target["expected_visual_elements"])
            logger.info(f"视觉标签筛选得到 {len(visual_segments)} 个候选片段")
            
            # 第二步：根据关键词筛选候选片段
            keyword_segments = self.repository.search_by_keywords(target["keywords"])
            logger.info(f"关键词筛选得到 {len(keyword_segments)} 个候选片段")
            
            # 第三步：计算候选片段的匹配分数
            candidate_scores = {}
            
            # 合并两个筛选结果
            all_segment_ids = set(visual_segments) | set(keyword_segments)
            
            for segment_id in all_segment_ids:
                # 计算匹配分数
                score = self.repository.compute_segment_match_score(segment_id, target, weights)
                candidate_scores[segment_id] = score
            
            # 第四步：排序并保留多样性
            top_candidates = self._select_diverse_candidates(
                candidate_scores, diversity_threshold, self.top_k)
            
            # 构建详细的匹配结果
            target_results = []
            
            for segment_id, score in top_candidates:
                segment = self.repository.get_segment(segment_id)
                if segment:
                    # 构建匹配详情
                    match_details = self._compute_match_details(segment, target)
                    
                    # 添加到结果
                    target_results.append({
                        "segment_id": segment_id,
                        "match_score": score,
                        "match_details": match_details,
                        "segment_info": {
                            "path": segment.get("path", ""),
                            "source_video": segment.get("source_video", ""),
                            "start_time": segment.get("start_time", 0),
                            "end_time": segment.get("end_time", 0),
                            "duration": segment.get("duration", 0),
                            "text_content": segment.get("text_content", "")
                        }
                    })
            
            matching_results[intent_type] = target_results
            logger.info(f"意图段落 {intent_type} 筛选出 {len(target_results)} 个候选片段")
        
        return matching_results
    
    def _filter_by_visual_tags(self, expected_elements, match_threshold=0.3):
        """
        根据期望的视觉元素筛选候选片段
        
        Args:
            expected_elements: 期望的视觉元素列表
            match_threshold: 匹配阈值，至少匹配的元素比例
            
        Returns:
            list: 匹配的候选片段ID列表
        """
        # 优先全部匹配
        segments = self.repository.search_by_visual_tags(expected_elements, match_all=True)
        
        # 如果结果太少，改用部分匹配
        if len(segments) < self.top_k:
            segments = self.repository.search_by_visual_tags(expected_elements, match_all=False)
            
            # 过滤掉匹配度过低的片段
            if expected_elements:
                filtered_segments = []
                for segment_id in segments:
                    segment = self.repository.get_segment(segment_id)
                    if segment:
                        # 计算匹配的视觉元素数量
                        visual_tags = [f"{tag.get('type')}:{tag.get('name')}" for tag in segment.get("visual_tags", [])]
                        matched_count = sum(1 for elem in expected_elements if elem in visual_tags)
                        match_ratio = matched_count / len(expected_elements)
                        
                        if match_ratio >= match_threshold:
                            filtered_segments.append(segment_id)
                
                segments = filtered_segments
        
        return segments
    
    def _select_diverse_candidates(self, candidate_scores, diversity_threshold, top_k):
        """
        从候选片段中选择多样性的结果
        
        Args:
            candidate_scores: 候选片段分数字典
            diversity_threshold: 多样性阈值
            top_k: 保留的结果数量
            
        Returns:
            list: 选中的候选片段ID和分数元组列表
        """
        if not candidate_scores:
            return []
            
        # 按分数排序
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 统计每个源视频的片段数量
        source_counter = defaultdict(int)
        selected_candidates = []
        
        for segment_id, score in sorted_candidates:
            segment = self.repository.get_segment(segment_id)
            if not segment:
                continue
                
            source_video = segment.get("source_video", "")
            
            # 检查是否超过多样性阈值
            if source_counter[source_video] < top_k * diversity_threshold:
                selected_candidates.append((segment_id, score))
                source_counter[source_video] += 1
            
            # 达到目标数量后停止
            if len(selected_candidates) >= top_k:
                break
        
        # 如果选择的候选片段不足，再次从排序列表中选择
        if len(selected_candidates) < top_k:
            for segment_id, score in sorted_candidates:
                if segment_id not in [c[0] for c in selected_candidates]:
                    selected_candidates.append((segment_id, score))
                    
                    if len(selected_candidates) >= top_k:
                        break
        
        return selected_candidates
    
    def _compute_match_details(self, segment, target):
        """
        计算匹配详情
        
        Args:
            segment: 片段数据
            target: 匹配目标
            
        Returns:
            dict: 匹配详情
        """
        # 视觉标签匹配
        visual_tags = [f"{tag.get('type')}:{tag.get('name')}" for tag in segment.get("visual_tags", [])]
        expected_elements = target.get("expected_visual_elements", [])
        
        matched_elements = [elem for elem in expected_elements if elem in visual_tags]
        visual_match_ratio = len(matched_elements) / len(expected_elements) if expected_elements else 0
        
        # 关键词匹配
        text_content = segment.get("text_content", "")
        keywords = target.get("keywords", [])
        
        matched_keywords = [keyword for keyword in keywords if keyword in text_content]
        keyword_match_ratio = len(matched_keywords) / len(keywords) if keywords else 0
        
        # 语义相似度（可选）
        semantic_similarity = 0.0
        
        return {
            "visual_match": {
                "matched_elements": matched_elements,
                "total_elements": len(expected_elements),
                "match_ratio": visual_match_ratio
            },
            "keyword_match": {
                "matched_keywords": matched_keywords,
                "total_keywords": len(keywords),
                "match_ratio": keyword_match_ratio
            },
            "semantic_similarity": semantic_similarity
        }
    
    def _save_matching_results(self, matching_results, output_path):
        """
        保存匹配结果
        
        Args:
            matching_results: 匹配结果
            output_path: 输出路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(matching_results, f, ensure_ascii=False, indent=2)
            logger.info(f"匹配结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存匹配结果失败: {str(e)}")


def run_matching(abstract_script_path=None, candidates_dir="data/processed/candidates", output_dir="data/processed", weights=None, diversity_threshold=0.3):
    """
    运行高效匹配流水线
    
    Args:
        abstract_script_path: 抽象脚本路径
        candidates_dir: 候选片段库目录
        output_dir: 输出目录
        weights: 匹配权重配置
        diversity_threshold: 多样性阈值
        
    Returns:
        str: 匹配结果路径
    """
    pipeline = MatchingPipeline(abstract_script_path, candidates_dir, output_dir)
    return pipeline.run(weights=weights, diversity_threshold=diversity_threshold)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行高效匹配流水线
    script_path = "data/processed/scripts/selected_abstract_script.json"
    matching_result = run_matching(abstract_script_path=script_path)
    print(f"匹配结果路径: {matching_result}") 