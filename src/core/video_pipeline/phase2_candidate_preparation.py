import os
import glob
import json
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.models.visual_analyzer import VisualAnalyzer
from src.core.utils.video_processor import VideoProcessor
from src.core.utils.text_processor import TextProcessor
from src.data_access.candidate_repository import CandidateRepository

logger = logging.getLogger(__name__)

class CandidatePreparationPipeline:
    """第二阶段处理流程 - 候选素材准备，实现候选视频片段库的构建"""
    
    def __init__(self, input_dir="data/test_samples/input/video", output_dir="data/processed"):
        """
        初始化候选素材准备流水线
        
        Args:
            input_dir: 输入视频目录
            output_dir: 输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        self.transcript_dir = os.path.join(output_dir, "transcripts")
        self.visual_analysis_dir = os.path.join(output_dir, "visual_analysis")
        self.segments_dir = os.path.join(output_dir, "segments")
        self.candidates_dir = os.path.join(output_dir, "candidates")
        
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.visual_analysis_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.candidates_dir, exist_ok=True)
        
        # 初始化处理器
        self.video_processor = VideoProcessor(temp_dir=os.path.join(output_dir, "temp"))
        self.visual_analyzer = VisualAnalyzer()
        self.text_processor = TextProcessor()
        
        # 初始化候选片段存储库
        self.repository = CandidateRepository(storage_dir=self.candidates_dir)
        
        logger.info(f"初始化候选素材准备流水线，输入目录: {input_dir}, 输出目录: {output_dir}")
    
    def run(self, max_workers=4, hot_word_id=None, sampling_rate=1.5, min_segment_duration=2.0, max_segment_duration=10.0):
        """
        运行候选素材准备流水线
        
        Args:
            max_workers: 并行处理的最大线程数
            hot_word_id: 热词ID，默认为None
            sampling_rate: 视觉分析采样率（秒）
            min_segment_duration: 最小片段时长（秒）
            max_segment_duration: 最大片段时长（秒）
            
        Returns:
            str: 候选片段库路径
        """
        logger.info("开始运行候选素材准备流水线")
        
        # 步骤2.1: 视频处理准备
        video_data = self._process_videos(max_workers, hot_word_id)
        
        if not video_data:
            logger.error("未能处理任何视频，流水线终止")
            return None
        
        # 步骤2.2: 视觉分析
        self._analyze_videos(video_data, max_workers, sampling_rate)
        
        # 步骤2.3: 片段切分
        segments = self._split_videos(video_data, min_segment_duration, max_segment_duration)
        
        # 步骤2.4: 候选库构建
        candidate_count = self._build_candidate_library(segments)
        
        logger.info(f"候选素材准备流水线运行完成，构建了 {candidate_count} 个候选片段")
        return self.candidates_dir
    
    def _process_videos(self, max_workers, hot_word_id):
        """
        并行处理多个视频，进行转录
        
        Args:
            max_workers: 并行处理的最大线程数
            hot_word_id: 热词ID
            
        Returns:
            list: 视频数据列表，每个元素包含视频路径、转录文本等
        """
        # 查找所有输入视频
        video_pattern = os.path.join(self.input_dir, "*.mp4")
        video_paths = glob.glob(video_pattern)
        
        if not video_paths:
            logger.warning(f"未找到匹配的视频文件: {video_pattern}")
            return []
        
        logger.info(f"找到 {len(video_paths)} 个候选视频文件待处理")
        
        # 并行处理视频
        video_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交转录任务
            future_to_video = {
                executor.submit(self._process_single_video, video_path, hot_word_id): video_path
                for video_path in video_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        video_data.append(result)
                        logger.info(f"候选视频处理成功: {video_path}")
                    else:
                        logger.error(f"候选视频处理失败: {video_path}")
                except Exception as e:
                    logger.error(f"候选视频处理异常: {video_path}, 错误: {str(e)}")
        
        logger.info(f"成功处理 {len(video_data)} 个候选视频")
        return video_data
    
    def _process_single_video(self, video_path, hot_word_id):
        """
        处理单个视频，包括转录和提取基本信息
        
        Args:
            video_path: 视频文件路径
            hot_word_id: 热词ID
            
        Returns:
            dict: 视频数据，包含视频路径、转录文本和基本信息
        """
        try:
            video_name = os.path.basename(video_path)
            video_name_noext = os.path.splitext(video_name)[0]
            
            # 转录路径
            transcript_path = os.path.join(self.transcript_dir, f"{video_name_noext}_transcript.json")
            
            # 1. 提取视频基本信息
            video_info = self.video_processor.extract_video_info(video_path)
            
            if not video_info:
                logger.error(f"提取视频信息失败: {video_path}")
                return None
            
            # 2. 进行视频转录
            transcript = self.video_processor.transcribe_video(
                video_path, 
                hot_word_id=hot_word_id,
                output_path=transcript_path
            )
            
            if not transcript:
                logger.error(f"视频转录失败: {video_path}")
                return None
            
            # 3. 提取纯文本内容
            full_text = self._extract_full_text(transcript)
            
            return {
                "video_path": video_path,
                "video_name": video_name,
                "transcript": transcript,
                "transcript_path": transcript_path,
                "video_info": video_info,
                "full_text": full_text
            }
            
        except Exception as e:
            logger.error(f"处理视频失败: {video_path}, 错误: {str(e)}")
            return None
    
    def _extract_full_text(self, transcript):
        """从转录结果中提取完整文本"""
        try:
            segments = transcript.get("segments", [])
            texts = [segment.get("text", "") for segment in segments]
            return " ".join(texts)
        except Exception as e:
            logger.error(f"提取完整文本失败: {str(e)}")
            return ""
    
    def _analyze_videos(self, video_data, max_workers, sampling_rate):
        """
        对所有视频进行视觉分析
        
        Args:
            video_data: 视频数据列表
            max_workers: 并行处理的最大线程数
            sampling_rate: 视觉分析采样率（秒）
        """
        logger.info("开始视觉分析")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交视觉分析任务
            future_to_video = {
                executor.submit(self._analyze_single_video, data, sampling_rate): data["video_path"]
                for data in video_data
            }
            
            # 收集结果
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(f"视觉分析成功: {video_path}")
                    else:
                        logger.error(f"视觉分析失败: {video_path}")
                except Exception as e:
                    logger.error(f"视觉分析异常: {video_path}, 错误: {str(e)}")
    
    def _analyze_single_video(self, video_data, sampling_rate):
        """
        对单个视频进行视觉分析
        
        Args:
            video_data: 视频数据
            sampling_rate: 采样率（秒）
            
        Returns:
            bool: 是否成功
        """
        try:
            video_path = video_data["video_path"]
            video_name = video_data["video_name"]
            video_name_noext = os.path.splitext(video_name)[0]
            
            # 视觉分析路径
            analysis_path = os.path.join(self.visual_analysis_dir, f"{video_name_noext}_visual_analysis.json")
            
            # 进行视觉分析
            analysis = self.visual_analyzer.analyze_video(video_path, sampling_rate=sampling_rate)
            
            if not analysis:
                logger.error(f"视觉分析失败: {video_path}")
                return False
            
            # 保存分析结果
            self.visual_analyzer.save_analysis(analysis, analysis_path)
            
            # 更新视频数据
            video_data["visual_analysis"] = analysis
            video_data["visual_analysis_path"] = analysis_path
            
            return True
            
        except Exception as e:
            logger.error(f"视觉分析失败: {video_path}, 错误: {str(e)}")
            return False
    
    def _split_videos(self, video_data, min_segment_duration, max_segment_duration):
        """
        根据转录和视觉分析结果切分视频
        
        Args:
            video_data: 视频数据列表
            min_segment_duration: 最小片段时长（秒）
            max_segment_duration: 最大片段时长（秒）
            
        Returns:
            list: 视频片段列表
        """
        logger.info("开始切分视频片段")
        
        all_segments = []
        
        for data in video_data:
            try:
                video_path = data["video_path"]
                video_name = data["video_name"]
                transcript = data["transcript"]
                visual_analysis = data.get("visual_analysis")
                
                if not visual_analysis:
                    logger.warning(f"跳过切分，视觉分析结果缺失: {video_path}")
                    continue
                
                # 根据转录结果分割段落
                transcript_segments = self._get_transcript_segments(transcript)
                
                # 调整段落时长，确保在min_segment_duration到max_segment_duration之间
                adjusted_segments = self._adjust_segment_durations(
                    transcript_segments, min_segment_duration, max_segment_duration)
                
                # 切分视频
                segment_paths = self.video_processor.split_video_segments(
                    video_path, adjusted_segments, 
                    output_dir=os.path.join(self.segments_dir, os.path.splitext(video_name)[0])
                )
                
                # 将视觉分析结果关联到每个片段
                segments_with_analysis = self._associate_visual_analysis(
                    segment_paths, visual_analysis, transcript, data["full_text"])
                
                all_segments.extend(segments_with_analysis)
                logger.info(f"切分完成: {video_path}, 生成 {len(segments_with_analysis)} 个片段")
                
            except Exception as e:
                logger.error(f"切分视频失败: {data['video_path']}, 错误: {str(e)}")
        
        logger.info(f"视频切分完成，共生成 {len(all_segments)} 个片段")
        return all_segments
    
    def _get_transcript_segments(self, transcript):
        """
        从转录结果中提取段落
        
        Args:
            transcript: 转录结果
            
        Returns:
            list: 段落列表，每个段落包含开始和结束时间
        """
        segments = []
        
        try:
            for i, segment in enumerate(transcript.get("segments", [])):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                
                if end_time <= start_time or not text.strip():
                    continue
                
                segments.append({
                    "index": i,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "text": text
                })
                
            return segments
        except Exception as e:
            logger.error(f"提取转录段落失败: {str(e)}")
            return []
    
    def _adjust_segment_durations(self, segments, min_duration, max_duration):
        """
        调整段落时长，确保在最小和最大时长之间
        
        Args:
            segments: 段落列表
            min_duration: 最小时长（秒）
            max_duration: 最大时长（秒）
            
        Returns:
            list: 调整后的段落列表
        """
        if not segments:
            return []
            
        adjusted_segments = []
        current_segment = None
        
        for segment in segments:
            # 如果当前段落为空，初始化为当前段落
            if current_segment is None:
                current_segment = segment.copy()
                continue
            
            current_duration = current_segment["end"] - current_segment["start"]
            
            # 如果当前段落时长已经达到最大时长，添加到结果并重置
            if current_duration >= max_duration:
                adjusted_segments.append(current_segment)
                current_segment = segment.copy()
                continue
            
            # 如果当前段落时长小于最小时长，尝试合并
            merged_duration = segment["end"] - current_segment["start"]
            
            if merged_duration <= max_duration:
                # 合并段落
                current_segment["end"] = segment["end"]
                current_segment["duration"] = current_segment["end"] - current_segment["start"]
                current_segment["text"] = current_segment["text"] + " " + segment["text"]
            else:
                # 添加当前段落并重置
                adjusted_segments.append(current_segment)
                current_segment = segment.copy()
        
        # 添加最后一个段落
        if current_segment is not None:
            adjusted_segments.append(current_segment)
        
        # 过滤掉时长过短的段落
        adjusted_segments = [s for s in adjusted_segments if s["duration"] >= min_duration]
        
        return adjusted_segments
    
    def _associate_visual_analysis(self, segment_paths, visual_analysis, transcript, full_text):
        """
        将视觉分析结果关联到每个片段
        
        Args:
            segment_paths: 片段路径列表
            visual_analysis: 视觉分析结果
            transcript: 转录结果
            full_text: 完整文本
            
        Returns:
            list: 关联了视觉分析的片段列表
        """
        if not segment_paths or not visual_analysis:
            return []
            
        timeline = visual_analysis.get("timeline", [])
        
        segments_with_analysis = []
        
        for segment_info in segment_paths:
            segment_id = str(uuid.uuid4())
            start_time = segment_info["start_time"]
            end_time = segment_info["end_time"]
            
            # 查找该时间范围内的视觉分析结果
            segment_timeline = [
                frame for frame in timeline
                if start_time <= frame.get("timestamp", 0) <= end_time
            ]
            
            # 提取视觉标签
            visual_tags = self._extract_visual_tags(segment_timeline)
            
            # 提取该时间范围内的文本
            segment_text = self._extract_segment_text(transcript, start_time, end_time)
            
            # 计算文本特征
            text_features = self.text_processor.compute_text_features(segment_text)
            
            # 构建完整的片段数据
            segment_data = {
                "segment_id": segment_id,
                "path": segment_info["path"],
                "source_video": segment_info["source_video"],
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "text_content": segment_text,
                "visual_tags": visual_tags,
                "text_features": text_features,
                "index": segment_info["index"]
            }
            
            segments_with_analysis.append(segment_data)
        
        return segments_with_analysis
    
    def _extract_visual_tags(self, timeline):
        """
        从时间线中提取视觉标签
        
        Args:
            timeline: 时间线数据
            
        Returns:
            list: 视觉标签列表
        """
        visual_tags = []
        
        # 物体标签
        objects = {}
        for frame in timeline:
            for obj in frame.get("objects", []):
                class_name = obj.get("class", "")
                if class_name:
                    if class_name not in objects:
                        objects[class_name] = 0
                    objects[class_name] += 1
        
        for obj, count in objects.items():
            if count >= len(timeline) * 0.3:  # 至少出现在30%的帧中
                visual_tags.append({
                    "type": "object",
                    "name": obj,
                    "confidence": count / len(timeline) if len(timeline) > 0 else 0
                })
        
        # 场景标签
        scenes = {}
        for frame in timeline:
            scene = frame.get("scene", "未知场景")
            if scene not in scenes:
                scenes[scene] = 0
            scenes[scene] += 1
        
        # 选择最多的场景
        if scenes:
            dominant_scene = max(scenes.items(), key=lambda x: x[1])
            visual_tags.append({
                "type": "scene",
                "name": dominant_scene[0],
                "confidence": dominant_scene[1] / len(timeline) if len(timeline) > 0 else 0
            })
        
        # 动作标签
        actions = {}
        for frame in timeline:
            action = frame.get("action", "未知动作")
            if action not in actions:
                actions[action] = 0
            actions[action] += 1
        
        # 选择最多的动作
        if actions:
            dominant_action = max(actions.items(), key=lambda x: x[1])
            visual_tags.append({
                "type": "action",
                "name": dominant_action[0],
                "confidence": dominant_action[1] / len(timeline) if len(timeline) > 0 else 0
            })
        
        # 产品展示标签
        displays = {}
        for frame in timeline:
            display = frame.get("product_display", "无产品展示")
            if display not in displays:
                displays[display] = 0
            displays[display] += 1
        
        # 选择最多的产品展示类型
        if displays:
            dominant_display = max(displays.items(), key=lambda x: x[1])
            if dominant_display[0] != "无产品展示":
                visual_tags.append({
                    "type": "product",
                    "name": dominant_display[0],
                    "confidence": dominant_display[1] / len(timeline) if len(timeline) > 0 else 0
                })
        
        return visual_tags
    
    def _extract_segment_text(self, transcript, start_time, end_time):
        """
        提取指定时间范围内的文本
        
        Args:
            transcript: 转录结果
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            str: 文本内容
        """
        segment_texts = []
        
        try:
            for segment in transcript.get("segments", []):
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", 0)
                text = segment.get("text", "")
                
                # 如果段落与时间范围有重叠
                if seg_end > start_time and seg_start < end_time:
                    segment_texts.append(text)
            
            return " ".join(segment_texts)
        except Exception as e:
            logger.error(f"提取片段文本失败: {str(e)}")
            return ""
    
    def _build_candidate_library(self, segments):
        """
        构建候选片段库
        
        Args:
            segments: 片段列表
            
        Returns:
            int: 添加的候选片段数量
        """
        logger.info("开始构建候选片段库")
        
        count = 0
        for segment in segments:
            if self.repository.add_segment(segment):
                count += 1
        
        # 保存候选片段库
        self.repository.save_data()
        
        logger.info(f"候选片段库构建完成，共 {count} 个候选片段")
        return count


def run_candidate_preparation(
    input_dir="data/test_samples/input/video", 
    output_dir="data/processed", 
    hot_word_id=None, 
    max_workers=4,
    sampling_rate=1.5,
    min_segment_duration=2.0, 
    max_segment_duration=10.0
):
    """
    运行候选素材准备流水线
    
    Args:
        input_dir: 输入视频目录
        output_dir: 输出目录
        hot_word_id: 热词ID
        max_workers: 并行处理的最大线程数
        sampling_rate: 视觉分析采样率（秒）
        min_segment_duration: 最小片段时长（秒）
        max_segment_duration: 最大片段时长（秒）
        
    Returns:
        str: 候选片段库路径
    """
    pipeline = CandidatePreparationPipeline(input_dir, output_dir)
    return pipeline.run(
        max_workers=max_workers,
        hot_word_id=hot_word_id,
        sampling_rate=sampling_rate,
        min_segment_duration=min_segment_duration,
        max_segment_duration=max_segment_duration
    )


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行候选素材准备流水线
    candidate_dir = run_candidate_preparation()
    print(f"候选片段库路径: {candidate_dir}") 