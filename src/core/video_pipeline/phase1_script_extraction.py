import os
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.core.models.bert_analyzer import BertIntentAnalyzer
from src.core.models.llm_analyzer import LLMScriptAnalyzer
from src.core.utils.video_processor import VideoProcessor
from src.core.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class ScriptExtractionPipeline:
    """第一阶段处理流程 - 并行脚本提炼，实现视频转录分析与抽象脚本生成"""
    
    def __init__(self, input_dir="data/input", output_dir="data/processed", hot_word_id="vocab-aivideo-4d73bdb1b5ef496d94f5104a957c012b"):
        """
        初始化并行脚本提炼流水线
        
        Args:
            input_dir: 输入视频目录
            output_dir: 输出目录
            hot_word_id: 热词ID
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.hot_word_id = hot_word_id
        
        # 创建输出目录
        self.transcript_dir = os.path.join(output_dir, "transcripts")
        self.video_info_dir = os.path.join(output_dir, "video_info")
        self.bert_analysis_dir = os.path.join(output_dir, "bert_analysis")
        self.llm_analysis_dir = os.path.join(output_dir, "llm_analysis")
        self.script_dir = os.path.join(output_dir, "scripts")
        
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.video_info_dir, exist_ok=True)
        os.makedirs(self.bert_analysis_dir, exist_ok=True)
        os.makedirs(self.llm_analysis_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)
        
        # 初始化处理器
        self.video_processor = VideoProcessor()
        self.bert_analyzer = BertIntentAnalyzer()
        self.llm_analyzer = LLMScriptAnalyzer()
        self.text_processor = TextProcessor()
        
        logger.info(f"初始化并行脚本提炼流水线，输入目录: {input_dir}, 输出目录: {output_dir}")
    
    def run(self, max_workers=4):
        """
        运行并行脚本提炼流水线
        
        Args:
            max_workers: 并行处理的最大线程数
            
        Returns:
            tuple: (BERT抽象脚本路径, LLM抽象脚本路径, 选择的脚本路径)
        """
        logger.info("开始运行并行脚本提炼流水线")
        
        # 步骤1.1: 视频转录预处理
        video_data = self._transcribe_videos(max_workers)
        
        if not video_data:
            logger.error("未能处理任何视频，流水线终止")
            return None, None, None
        
        # 步骤1.2: BERT分析路径
        bert_script_path = self._run_bert_analysis(video_data)
        
        # 步骤1.3: LLM分析路径
        llm_script_path = self._run_llm_analysis(video_data)
        
        # 步骤1.4: 结果对比与选择
        selected_script_path = self._compare_and_select_script(bert_script_path, llm_script_path)
        
        logger.info("并行脚本提炼流水线运行完成")
        return bert_script_path, llm_script_path, selected_script_path
    
    def _transcribe_videos(self, max_workers):
        """
        并行转录多个视频
        
        Args:
            max_workers: 并行处理的最大线程数
            
        Returns:
            list: 视频数据列表，每个元素包含视频路径、转录文本和基本信息
        """
        # 查找所有输入视频
        video_pattern = os.path.join(self.input_dir, "test*.mp4")
        video_paths = glob.glob(video_pattern)
        
        if not video_paths:
            logger.warning(f"未找到匹配的视频文件: {video_pattern}")
            return []
        
        logger.info(f"找到 {len(video_paths)} 个视频文件待处理")
        
        # 并行处理视频
        video_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交转录任务
            future_to_video = {
                executor.submit(self._process_single_video, video_path): video_path
                for video_path in video_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        video_data.append(result)
                        logger.info(f"视频处理成功: {video_path}")
                    else:
                        logger.error(f"视频处理失败: {video_path}")
                except Exception as e:
                    logger.error(f"视频处理异常: {video_path}, 错误: {str(e)}")
        
        logger.info(f"成功处理 {len(video_data)} 个视频")
        return video_data
    
    def _process_single_video(self, video_path):
        """
        处理单个视频，包括转录和提取基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 视频数据，包含视频路径、转录文本和基本信息
        """
        try:
            video_name = os.path.basename(video_path)
            video_name_noext = os.path.splitext(video_name)[0]
            
            # 转录路径
            transcript_path = os.path.join(self.transcript_dir, f"{video_name_noext}_transcript.json")
            
            # 视频信息路径
            video_info_path = os.path.join(self.video_info_dir, f"{video_name_noext}_info.json")
            
            # 1. 提取视频基本信息
            video_info = self.video_processor.extract_video_info(video_path)
            
            if not video_info:
                logger.error(f"提取视频信息失败: {video_path}")
                return None
            
            # 保存视频信息
            with open(video_info_path, 'w', encoding='utf-8') as f:
                json.dump(video_info, f, ensure_ascii=False, indent=2)
            
            # 2. 进行视频转录
            transcript = self.video_processor.transcribe_video(
                video_path, 
                hot_word_id=self.hot_word_id,
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
                "video_info_path": video_info_path,
                "full_text": full_text
            }
            
        except Exception as e:
            logger.error(f"处理视频失败: {video_path}, 错误: {str(e)}")
            return None
    
    def _extract_full_text(self, transcript):
        """
        从转录结果中提取完整文本
        
        Args:
            transcript: 转录结果JSON对象
            
        Returns:
            str: 完整文本
        """
        try:
            segments = transcript.get("segments", [])
            texts = [segment.get("text", "") for segment in segments]
            return " ".join(texts)
        except Exception as e:
            logger.error(f"提取完整文本失败: {str(e)}")
            return ""
    
    def _run_bert_analysis(self, video_data):
        """
        对所有视频运行BERT意图分析
        
        Args:
            video_data: 视频数据列表
            
        Returns:
            str: BERT抽象脚本路径
        """
        logger.info("开始BERT意图分析")
        
        # 分析每个视频
        video_analyses = []
        
        for data in video_data:
            video_name = data["video_name"]
            video_name_noext = os.path.splitext(video_name)[0]
            transcript = data["transcript"]
            
            # 分析路径
            analysis_path = os.path.join(self.bert_analysis_dir, f"{video_name_noext}_bert_analysis.json")
            
            # 进行BERT分析
            try:
                # 分析转录文本中的意图和关键词
                analysis_result = self.bert_analyzer.analyze_transcript(transcript)
                
                # 保存分析结果
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, ensure_ascii=False, indent=2)
                
                video_analyses.append(analysis_result)
                logger.info(f"BERT分析完成: {video_name}")
            except Exception as e:
                logger.error(f"BERT分析失败: {video_name}, 错误: {str(e)}")
        
        # 生成BERT抽象脚本
        bert_script_path = os.path.join(self.script_dir, "bert_abstract_script.json")
        
        try:
            # 生成抽象脚本
            abstract_script = self.bert_analyzer.generate_abstract_script(video_analyses)
            
            # 保存抽象脚本
            self.bert_analyzer.save_abstract_script(abstract_script, bert_script_path)
            logger.info(f"BERT抽象脚本已生成: {bert_script_path}")
            
            return bert_script_path
        except Exception as e:
            logger.error(f"生成BERT抽象脚本失败: {str(e)}")
            return None
    
    def _run_llm_analysis(self, video_data):
        """
        对所有视频运行LLM分析
        
        Args:
            video_data: 视频数据列表
            
        Returns:
            str: LLM抽象脚本路径
        """
        logger.info("开始LLM意图分析")
        
        # 收集所有视频的转录文本
        transcripts = []
        
        for data in video_data:
            transcripts.append(data["full_text"])
        
        # 进行LLM分析
        llm_script_path = os.path.join(self.script_dir, "llm_abstract_script.json")
        
        try:
            # 分析所有转录文本
            abstract_script = self.llm_analyzer.analyze_transcripts(transcripts)
            
            # 保存抽象脚本
            self.llm_analyzer.save_abstract_script(abstract_script, llm_script_path)
            logger.info(f"LLM抽象脚本已生成: {llm_script_path}")
            
            return llm_script_path
        except Exception as e:
            logger.error(f"生成LLM抽象脚本失败: {str(e)}")
            return None
    
    def _compare_and_select_script(self, bert_script_path, llm_script_path):
        """
        比较BERT和LLM生成的抽象脚本，选择一个作为后续阶段的输入
        
        Args:
            bert_script_path: BERT抽象脚本路径
            llm_script_path: LLM抽象脚本路径
            
        Returns:
            str: 选择的脚本路径
        """
        if not bert_script_path and not llm_script_path:
            logger.error("BERT和LLM抽象脚本均生成失败")
            return None
        
        if not bert_script_path:
            logger.warning("BERT抽象脚本生成失败，使用LLM抽象脚本")
            return llm_script_path
        
        if not llm_script_path:
            logger.warning("LLM抽象脚本生成失败，使用BERT抽象脚本")
            return bert_script_path
        
        # 加载两个脚本
        try:
            with open(bert_script_path, 'r', encoding='utf-8') as f:
                bert_script = json.load(f)
                
            with open(llm_script_path, 'r', encoding='utf-8') as f:
                llm_script = json.load(f)
                
            # 比较两个脚本
            comparison = self.llm_analyzer.compare_scripts(bert_script, llm_script)
            
            # 保存比较结果
            comparison_path = os.path.join(self.script_dir, "script_comparison.json")
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            
            # 根据比较结果选择脚本
            selected_script = comparison.get("recommendation", "llm")
            selected_path = llm_script_path if selected_script == "llm" else bert_script_path
            
            # 创建选择的脚本的副本
            selected_script_path = os.path.join(self.script_dir, "selected_abstract_script.json")
            selected_script_obj = llm_script if selected_script == "llm" else bert_script
            
            with open(selected_script_path, 'w', encoding='utf-8') as f:
                json.dump(selected_script_obj, f, ensure_ascii=False, indent=2)
            
            logger.info(f"脚本比较完成，选择了 {selected_script} 脚本: {selected_script_path}")
            return selected_script_path
            
        except Exception as e:
            logger.error(f"比较脚本失败: {str(e)}")
            # 默认使用LLM脚本
            return llm_script_path


def run_script_extraction(input_dir="data/input", output_dir="data/processed", hot_word_id="vocab-aivideo-4d73bdb1b5ef496d94f5104a957c012b", max_workers=4):
    """
    运行并行脚本提炼流水线
    
    Args:
        input_dir: 输入视频目录
        output_dir: 输出目录
        hot_word_id: 热词ID
        max_workers: 并行处理的最大线程数
        
    Returns:
        str: 选择的脚本路径
    """
    pipeline = ScriptExtractionPipeline(input_dir, output_dir, hot_word_id)
    _, _, selected_script_path = pipeline.run(max_workers)
    return selected_script_path


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行脚本提取
    script_path = run_script_extraction()
    print(f"选择的抽象脚本路径: {script_path}") 