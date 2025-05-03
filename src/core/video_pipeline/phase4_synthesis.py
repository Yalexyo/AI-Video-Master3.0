import os
import json
import logging
from collections import defaultdict

from src.core.utils.video_processor import VideoProcessor
from src.core.utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class SynthesisPipeline:
    """第四阶段处理流程 - 音频驱动视频合成，实现最终视频生成"""
    
    def __init__(self, matching_results_path=None, reference_video_path=None, output_dir="data/processed"):
        """
        初始化音频驱动视频合成流水线
        
        Args:
            matching_results_path: 匹配结果路径，默认为None
            reference_video_path: 参考视频路径，默认为None
            output_dir: 输出目录
        """
        self.matching_results_path = matching_results_path
        self.reference_video_path = reference_video_path
        self.output_dir = output_dir
        
        # 创建输出目录
        self.audio_dir = os.path.join(output_dir, "audio")
        self.segments_dir = os.path.join(output_dir, "synthesis_segments")
        self.output_video_dir = os.path.join(output_dir, "output", "videos")
        
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.output_video_dir, exist_ok=True)
        
        # 初始化处理器
        self.video_processor = VideoProcessor(temp_dir=os.path.join(output_dir, "temp"))
        self.audio_processor = AudioProcessor(temp_dir=os.path.join(output_dir, "temp", "audio"))
        
        logger.info(f"初始化音频驱动视频合成流水线，匹配结果路径: {matching_results_path}, 参考视频路径: {reference_video_path}")
    
    def run(self, matching_results_path=None, reference_video_path=None, output_video_name="synthesized_video.mp4", transitions=None):
        """
        运行音频驱动视频合成流水线
        
        Args:
            matching_results_path: 匹配结果路径，默认为None
            reference_video_path: 参考视频路径，默认为None
            output_video_name: 输出视频名称，默认为"synthesized_video.mp4"
            transitions: 转场效果配置，默认为None
            
        Returns:
            str: 输出视频路径
        """
        # 使用传入的路径或初始化时设置的路径
        match_path = matching_results_path or self.matching_results_path
        ref_video_path = reference_video_path or self.reference_video_path
        
        if not match_path:
            logger.error("未指定匹配结果路径")
            return None
            
        if not ref_video_path:
            logger.error("未指定参考视频路径")
            return None
        
        logger.info("开始运行音频驱动视频合成流水线")
        
        # 步骤4.1: 音频分析与提取
        audio_data = self._analyze_reference_audio(ref_video_path)
        
        if not audio_data:
            logger.error("音频分析失败，无法继续合成")
            return None
        
        # 步骤4.2: 音频驱动的片段选择
        selected_segments = self._select_segments_by_audio(audio_data, match_path)
        
        # 步骤4.3: 视觉统一与剪辑
        uniform_segments = self._unify_segments(selected_segments, ref_video_path)
        
        # 步骤4.4: 无缝填充策略
        filled_segments = self._fill_gaps(uniform_segments, audio_data)
        
        # 步骤4.5: 最终合成与输出
        output_path = os.path.join(self.output_video_dir, output_video_name)
        success = self._synthesize_final_video(filled_segments, audio_data["full_audio_path"], output_path, transitions)
        
        if not success:
            logger.error("视频合成失败")
            return None
            
        logger.info(f"音频驱动视频合成流水线运行完成，输出视频: {output_path}")
        return output_path
    
    def _analyze_reference_audio(self, video_path):
        """
        分析参考视频的音频
        
        Args:
            video_path: 视频路径
            
        Returns:
            dict: 音频数据，包含音频路径、段落信息等
        """
        logger.info(f"开始分析参考视频音频: {video_path}")
        
        # 提取音频
        video_name = os.path.basename(video_path)
        video_name_noext = os.path.splitext(video_name)[0]
        
        audio_path = os.path.join(self.audio_dir, f"{video_name_noext}_audio.wav")
        extracted_audio = self.video_processor.extract_audio(video_path, output_path=audio_path)
        
        if not extracted_audio:
            logger.error(f"从视频提取音频失败: {video_path}")
            return None
        
        try:
            # 提取视频信息
            video_info = self.video_processor.extract_video_info(video_path)
            
            if not video_info:
                logger.error(f"提取视频信息失败: {video_path}")
                return None
            
            # 分析音频，识别语音段落
            speech_segments = self.audio_processor.detect_speech_segments(audio_path)
            
            if not speech_segments:
                logger.warning(f"未检测到语音段落: {audio_path}")
                # 创建一个覆盖整个视频的默认段落
                speech_segments = [{
                    "index": 0,
                    "start": 0,
                    "end": video_info["duration"],
                    "duration": video_info["duration"]
                }]
            
            # 检测节奏点
            rhythm_points = self.audio_processor.detect_audio_rhythm(audio_path)
            
            # 提取音频特征
            audio_features = self.audio_processor.extract_audio_features(audio_path)
            
            # 将语音段落与传入的匹配目标对应
            audio_segments = self._align_segments_with_intents(speech_segments)
            
            # 构建音频数据
            audio_data = {
                "video_path": video_path,
                "full_audio_path": audio_path,
                "video_info": video_info,
                "speech_segments": speech_segments,
                "audio_segments": audio_segments,
                "rhythm_points": rhythm_points,
                "audio_features": audio_features
            }
            
            logger.info(f"音频分析完成，识别到 {len(speech_segments)} 个语音段落")
            return audio_data
            
        except Exception as e:
            logger.error(f"音频分析失败: {str(e)}")
            return None
    
    def _align_segments_with_intents(self, speech_segments):
        """
        将语音段落与意图进行对齐
        
        Args:
            speech_segments: 语音段落列表
            
        Returns:
            list: 对齐后的段落列表，每个段落包含意图类型
        """
        # 这里采用简单的均匀分配策略
        # 在实际应用中，可能需要更复杂的算法来匹配语音内容与意图
        
        # 意图类型列表
        intent_types = ["问题引入", "产品介绍", "效果展示", "促销信息"]
        
        # 计算每个意图类型的段落数量
        total_segments = len(speech_segments)
        segments_per_intent = max(1, total_segments // len(intent_types))
        
        # 分配意图类型
        audio_segments = []
        
        for i, segment in enumerate(speech_segments):
            intent_index = min(i // segments_per_intent, len(intent_types) - 1)
            intent_type = intent_types[intent_index]
            
            # 复制段落并添加意图类型
            audio_segment = segment.copy()
            audio_segment["intent_type"] = intent_type
            audio_segments.append(audio_segment)
        
        return audio_segments
    
    def _select_segments_by_audio(self, audio_data, matching_results_path):
        """
        根据音频段落选择匹配的视频片段
        
        Args:
            audio_data: 音频数据
            matching_results_path: 匹配结果路径
            
        Returns:
            list: 选择的片段列表
        """
        logger.info("开始根据音频段落选择匹配的视频片段")
        
        try:
            # 加载匹配结果
            with open(matching_results_path, 'r', encoding='utf-8') as f:
                matching_results = json.load(f)
                
            if not matching_results:
                logger.error("匹配结果为空")
                return []
            
            # 获取音频段落
            audio_segments = audio_data.get("audio_segments", [])
            
            if not audio_segments:
                logger.error("音频段落为空")
                return []
            
            # 为每个音频段落选择匹配的视频片段
            selected_segments = []
            
            for segment in audio_segments:
                intent_type = segment.get("intent_type")
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                duration = end_time - start_time
                
                # 获取该意图类型的匹配片段
                intent_candidates = matching_results.get(intent_type, [])
                
                if not intent_candidates:
                    logger.warning(f"意图类型 {intent_type} 没有匹配的片段")
                    continue
                
                # 选择匹配的片段
                selected_candidate = self._select_best_candidate(intent_candidates, duration, selected_segments)
                
                if selected_candidate:
                    # 添加音频段落信息
                    candidate_with_audio = {
                        "segment_id": selected_candidate.get("segment_id"),
                        "segment_info": selected_candidate.get("segment_info", {}),
                        "match_score": selected_candidate.get("match_score", 0),
                        "audio_segment": {
                            "start": start_time,
                            "end": end_time,
                            "duration": duration,
                            "intent_type": intent_type
                        }
                    }
                    
                    selected_segments.append(candidate_with_audio)
                    logger.info(f"为意图 {intent_type} 选择了片段 {selected_candidate.get('segment_id')}")
                else:
                    logger.warning(f"无法为意图 {intent_type} 选择合适的片段")
            
            logger.info(f"片段选择完成，共选择了 {len(selected_segments)} 个片段")
            return selected_segments
            
        except Exception as e:
            logger.error(f"片段选择失败: {str(e)}")
            return []
    
    def _select_best_candidate(self, candidates, target_duration, already_selected):
        """
        从候选片段中选择最佳匹配的片段
        
        Args:
            candidates: 候选片段列表
            target_duration: 目标时长
            already_selected: 已选择的片段列表
            
        Returns:
            dict: 选择的片段，如果没有合适的片段则返回None
        """
        if not candidates:
            return None
            
        # 已选择的源视频
        selected_sources = set()
        for selected in already_selected:
            source_video = selected.get("segment_info", {}).get("source_video", "")
            if source_video:
                selected_sources.add(source_video)
        
        # 按照匹配分数降序排序
        sorted_candidates = sorted(candidates, key=lambda x: x.get("match_score", 0), reverse=True)
        
        # 尝试选择最佳候选片段
        best_candidate = None
        min_duration_diff = float('inf')
        
        for candidate in sorted_candidates:
            segment_info = candidate.get("segment_info", {})
            source_video = segment_info.get("source_video", "")
            segment_duration = segment_info.get("duration", 0)
            
            # 计算时长差异
            duration_diff = abs(segment_duration - target_duration)
            
            # 如果这个片段来自新源视频，或者时长差异比当前最小值小，选择它
            if source_video not in selected_sources or duration_diff < min_duration_diff:
                best_candidate = candidate
                min_duration_diff = duration_diff
                
                # 如果时长差异足够小，直接选择
                if duration_diff < 1.0:  # 时长差异小于1秒
                    break
        
        return best_candidate
    
    def _unify_segments(self, selected_segments, reference_video_path):
        """
        将选择的片段统一为与参考视频相同的格式
        
        Args:
            selected_segments: 选择的片段列表
            reference_video_path: 参考视频路径
            
        Returns:
            list: 统一后的片段列表
        """
        logger.info("开始将选择的片段统一为与参考视频相同的格式")
        
        try:
            # 提取参考视频信息
            ref_info = self.video_processor.extract_video_info(reference_video_path)
            
            if not ref_info:
                logger.error(f"提取参考视频信息失败: {reference_video_path}")
                return selected_segments
            
            ref_width = ref_info.get("width", 1920)
            ref_height = ref_info.get("height", 1080)
            ref_fps = ref_info.get("fps", 30)
            
            # 统一片段
            uniform_segments = []
            
            for i, segment in enumerate(selected_segments):
                segment_info = segment.get("segment_info", {})
                segment_path = segment_info.get("path", "")
                
                if not segment_path or not os.path.exists(segment_path):
                    logger.warning(f"片段路径不存在: {segment_path}")
                    continue
                
                # 提取片段信息
                segment_info = self.video_processor.extract_video_info(segment_path)
                
                if not segment_info:
                    logger.warning(f"提取片段信息失败: {segment_path}")
                    continue
                
                # 如果需要统一分辨率
                if segment_info.get("width") != ref_width or segment_info.get("height") != ref_height:
                    # 统一分辨率
                    uniform_path = os.path.join(
                        self.segments_dir, 
                        f"uniform_{i:03d}_{os.path.basename(segment_path)}"
                    )
                    
                    success = self.video_processor.resize_video(
                        segment_path, uniform_path, ref_width, ref_height)
                    
                    if not success:
                        logger.warning(f"调整片段分辨率失败: {segment_path}")
                        continue
                    
                    # 更新片段路径
                    segment["uniform_path"] = uniform_path
                else:
                    # 不需要调整分辨率
                    segment["uniform_path"] = segment_path
                
                # 添加参考信息
                segment["reference_info"] = {
                    "width": ref_width,
                    "height": ref_height,
                    "fps": ref_fps
                }
                
                uniform_segments.append(segment)
            
            logger.info(f"片段统一完成，共处理了 {len(uniform_segments)} 个片段")
            return uniform_segments
            
        except Exception as e:
            logger.error(f"片段统一失败: {str(e)}")
            return selected_segments
    
    def _fill_gaps(self, segments, audio_data):
        """
        填充片段之间的空白，确保无缝覆盖整个音频
        
        Args:
            segments: 片段列表
            audio_data: 音频数据
            
        Returns:
            list: 填充后的片段列表
        """
        logger.info("开始填充片段之间的空白")
        
        if not segments:
            logger.warning("没有片段需要填充")
            return []
        
        try:
            # 计算片段覆盖情况
            audio_segments = []
            
            for segment in segments:
                audio_segment = segment.get("audio_segment", {})
                start_time = audio_segment.get("start", 0)
                end_time = audio_segment.get("end", 0)
                
                audio_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "segment": segment
                })
            
            # 按开始时间排序
            audio_segments.sort(key=lambda x: x["start"])
            
            # 检测空白并填充
            filled_segments = []
            current_end = 0
            
            for i, segment_data in enumerate(audio_segments):
                start_time = segment_data["start"]
                end_time = segment_data["end"]
                segment = segment_data["segment"]
                
                # 如果当前段落与前一个段落之间有空白，填充它
                if start_time > current_end:
                    gap_duration = start_time - current_end
                    logger.info(f"检测到空白: {current_end} - {start_time}, 持续 {gap_duration:.2f} 秒")
                    
                    # 填充策略
                    if gap_duration < 1.0:
                        # 对于短空白，延长前一个片段
                        if filled_segments:
                            prev_segment = filled_segments[-1]
                            prev_segment["audio_segment"]["end"] = start_time
                            prev_segment["audio_segment"]["duration"] = prev_segment["audio_segment"]["end"] - prev_segment["audio_segment"]["start"]
                            logger.info(f"通过延长前一个片段填充短空白")
                    else:
                        # 对于长空白，插入相关B-roll画面
                        gap_filler = self._create_gap_filler(current_end, start_time, segments)
                        
                        if gap_filler:
                            filled_segments.append(gap_filler)
                            logger.info(f"使用B-roll填充长空白")
                
                # 添加当前片段
                filled_segments.append(segment)
                current_end = end_time
            
            # 检查是否覆盖整个音频
            full_audio_duration = audio_data.get("video_info", {}).get("duration", 0)
            
            if current_end < full_audio_duration:
                # 如果末尾有空白，填充它
                gap_duration = full_audio_duration - current_end
                logger.info(f"检测到末尾空白: {current_end} - {full_audio_duration}, 持续 {gap_duration:.2f} 秒")
                
                # 填充策略
                if gap_duration < 1.0:
                    # 对于短空白，延长最后一个片段
                    if filled_segments:
                        last_segment = filled_segments[-1]
                        last_segment["audio_segment"]["end"] = full_audio_duration
                        last_segment["audio_segment"]["duration"] = last_segment["audio_segment"]["end"] - last_segment["audio_segment"]["start"]
                        logger.info(f"通过延长最后一个片段填充末尾短空白")
                else:
                    # 对于长空白，插入相关B-roll画面
                    gap_filler = self._create_gap_filler(current_end, full_audio_duration, segments)
                    
                    if gap_filler:
                        filled_segments.append(gap_filler)
                        logger.info(f"使用B-roll填充末尾长空白")
            
            logger.info(f"空白填充完成，共 {len(filled_segments)} 个片段")
            return filled_segments
            
        except Exception as e:
            logger.error(f"空白填充失败: {str(e)}")
            return segments
    
    def _create_gap_filler(self, start_time, end_time, segments):
        """
        创建用于填充空白的片段
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            segments: 现有片段列表
            
        Returns:
            dict: 填充片段
        """
        # 选择一个适合的片段作为填充
        # 策略：优先选择产品特写类型的片段
        
        suitable_segments = []
        
        for segment in segments:
            segment_info = segment.get("segment_info", {})
            source_video = segment_info.get("source_video", "")
            path = segment.get("uniform_path", "")
            
            if not path or not os.path.exists(path):
                continue
            
            # 确认是否为产品展示类型
            intent_type = segment.get("audio_segment", {}).get("intent_type", "")
            if intent_type == "产品介绍" or intent_type == "促销信息":
                suitable_segments.append(segment)
        
        if not suitable_segments:
            # 如果没有合适的片段，使用任何可用片段
            suitable_segments = [s for s in segments if s.get("uniform_path") and os.path.exists(s.get("uniform_path", ""))]
        
        if not suitable_segments:
            logger.warning("没有合适的片段用于填充空白")
            return None
        
        # 随机选择一个片段
        import random
        filler_segment = random.choice(suitable_segments)
        
        # 复制片段信息，更新时间范围
        gap_filler = filler_segment.copy()
        gap_filler["audio_segment"] = {
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "intent_type": "填充片段"
        }
        
        return gap_filler
    
    def _synthesize_final_video(self, segments, audio_path, output_path, transitions=None):
        """
        合成最终视频
        
        Args:
            segments: 片段列表
            audio_path: 音频路径
            output_path: 输出视频路径
            transitions: 转场效果配置
            
        Returns:
            bool: 是否成功
        """
        logger.info("开始合成最终视频")
        
        if not segments:
            logger.error("没有片段可用于合成")
            return False
        
        try:
            # 按音频段落开始时间排序片段
            sorted_segments = sorted(
                segments, 
                key=lambda x: x.get("audio_segment", {}).get("start", 0)
            )
            
            # 准备临时片段，调整时长以匹配音频段落
            temp_segments = []
            
            for i, segment in enumerate(sorted_segments):
                # 获取相关信息
                uniform_path = segment.get("uniform_path", "")
                audio_segment = segment.get("audio_segment", {})
                audio_duration = audio_segment.get("duration", 0)
                
                if not uniform_path or not os.path.exists(uniform_path):
                    logger.warning(f"片段路径不存在: {uniform_path}")
                    continue
                
                # 提取片段信息
                segment_info = self.video_processor.extract_video_info(uniform_path)
                
                if not segment_info:
                    logger.warning(f"提取片段信息失败: {uniform_path}")
                    continue
                
                segment_duration = segment_info.get("duration", 0)
                
                # 调整片段时长以匹配音频段落
                if abs(segment_duration - audio_duration) > 0.1:  # 如果时长差异大于0.1秒
                    # 创建临时片段
                    temp_path = os.path.join(
                        self.segments_dir, 
                        f"temp_{i:03d}_{os.path.basename(uniform_path)}"
                    )
                    
                    # 调整时长策略
                    if segment_duration > audio_duration:
                        # 如果片段过长，裁剪它
                        clip = self.video_processor.split_video_segments(
                            uniform_path, 
                            [{
                                "start": 0,
                                "end": audio_duration,
                                "duration": audio_duration
                            }],
                            output_dir=os.path.dirname(temp_path)
                        )
                        
                        if clip and clip[0]:
                            temp_path = clip[0].get("path", "")
                        else:
                            logger.warning(f"裁剪片段失败: {uniform_path}")
                            temp_path = uniform_path
                    else:
                        # 如果片段过短，通过速度调整来延长它
                        speed_factor = segment_duration / audio_duration
                        self.video_processor.apply_speed_adjustment(
                            uniform_path, temp_path, speed_factor
                        )
                        
                        if not os.path.exists(temp_path):
                            logger.warning(f"调整片段速度失败: {uniform_path}")
                            temp_path = uniform_path
                else:
                    # 时长一致，不需要调整
                    temp_path = uniform_path
                
                # 添加到临时片段列表
                temp_segments.append(temp_path)
            
            # 合并所有片段
            if not temp_segments:
                logger.error("没有有效的临时片段可用于合成")
                return False
            
            # 合并视频片段
            merged_path = os.path.join(self.segments_dir, "merged_video.mp4")
            merge_success = self.video_processor.merge_video_segments(
                temp_segments, merged_path, transitions=transitions
            )
            
            if not merge_success:
                logger.error("合并视频片段失败")
                return False
            
            # 替换音频
            replace_success = self.video_processor.replace_audio(
                merged_path, audio_path, output_path
            )
            
            if not replace_success:
                logger.error("替换音频失败")
                return False
            
            logger.info(f"视频合成成功，输出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"视频合成失败: {str(e)}")
            return False


def run_synthesis(matching_results_path=None, reference_video_path="data/input/test05.mp4", output_dir="data/processed", output_video_name="synthesized_video.mp4", transitions=None):
    """
    运行音频驱动视频合成流水线
    
    Args:
        matching_results_path: 匹配结果路径
        reference_video_path: 参考视频路径
        output_dir: 输出目录
        output_video_name: 输出视频名称
        transitions: 转场效果配置
        
    Returns:
        str: 输出视频路径
    """
    pipeline = SynthesisPipeline(matching_results_path, reference_video_path, output_dir)
    return pipeline.run(output_video_name=output_video_name, transitions=transitions)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行音频驱动视频合成流水线
    matching_path = "data/processed/matching/matching_results.json"
    reference_path = "data/input/test05.mp4"
    output_video = run_synthesis(matching_results_path=matching_path, reference_video_path=reference_path)
    print(f"输出视频路径: {output_video}") 