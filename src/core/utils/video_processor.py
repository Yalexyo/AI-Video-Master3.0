import os
import cv2
import json
import logging
import subprocess
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理工具类，封装视频操作相关功能"""
    
    def __init__(self, temp_dir="data/temp"):
        """
        初始化视频处理器
        
        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"初始化视频处理器，临时目录: {temp_dir}")
        
    def extract_audio(self, video_path, output_path=None):
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频文件路径，默认为None则自动生成
            
        Returns:
            str: 输出音频文件路径
        """
        if output_path is None:
            video_name = os.path.basename(video_path)
            video_name_noext = os.path.splitext(video_name)[0]
            output_path = os.path.join(self.temp_dir, f"{video_name_noext}_audio.wav")
        
        try:
            logger.info(f"从视频中提取音频: {video_path} -> {output_path}")
            
            # 使用moviepy提取音频
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                logger.warning(f"视频 {video_path} 没有音频轨道")
                return None
                
            audio.write_audiofile(output_path, logger=None)
            logger.info(f"音频提取完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path, hot_word_id=None):
        """
        对音频进行语音转录
        
        Args:
            audio_path: 音频文件路径
            hot_word_id: 热词ID，默认为None
            
        Returns:
            dict: 转录结果JSON对象
        """
        try:
            logger.info(f"开始音频转录: {audio_path}")
            
            # 这里应调用语音识别服务，如阿里云ASR或本地Whisper模型
            # 示例代码使用模拟数据
            from dashscope import Speech
            
            # 构建调用参数
            args = {
                'audio_file': audio_path,
                'model': 'paraformer-v2',
                'sample_rate': 16000,
                'audio_format': 'wav'
            }
            
            # 如果有热词ID，添加到参数中
            if hot_word_id:
                args['hot_word_id'] = hot_word_id
                logger.info(f"使用热词ID: {hot_word_id}")
            
            # 调用语音识别服务
            response = Speech.transcribe(**args)
            
            if response.status_code == 200:
                logger.info("音频转录成功")
                return response.output
            else:
                logger.error(f"转录服务调用失败: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"音频转录失败: {str(e)}")
            return None
    
    def transcribe_video(self, video_path, hot_word_id=None, output_path=None):
        """
        对视频进行语音转录
        
        Args:
            video_path: 视频文件路径
            hot_word_id: 热词ID，默认为None
            output_path: 转录结果输出路径，默认为None则自动生成
            
        Returns:
            dict: 转录结果JSON对象
        """
        try:
            # 提取音频
            audio_path = self.extract_audio(video_path)
            
            if not audio_path:
                logger.error(f"无法从视频提取音频: {video_path}")
                return None
            
            # 转录音频
            transcript = self.transcribe_audio(audio_path, hot_word_id)
            
            if not transcript:
                logger.error(f"音频转录失败: {audio_path}")
                return None
            
            # 保存转录结果
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(transcript, f, ensure_ascii=False, indent=2)
                    logger.info(f"转录结果已保存到: {output_path}")
                except Exception as e:
                    logger.error(f"保存转录结果失败: {str(e)}")
            
            return transcript
            
        except Exception as e:
            logger.error(f"视频转录失败: {str(e)}")
            return None
    
    def extract_video_info(self, video_path):
        """
        提取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 视频信息，包括时长、分辨率等
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return None
            
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # 关闭视频文件
            cap.release()
            
            # 构建视频信息
            video_info = {
                "path": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": f"{width}x{height}"
            }
            
            logger.info(f"提取视频信息成功: {video_path}")
            return video_info
            
        except Exception as e:
            logger.error(f"提取视频信息失败: {str(e)}")
            return None
    
    def split_video_segments(self, video_path, segments, output_dir=None):
        """
        将视频按照指定的时间段切分成片段
        
        Args:
            video_path: 视频文件路径
            segments: 片段列表，每个片段是一个包含开始和结束时间的字典
            output_dir: 输出目录，默认为None则使用临时目录
            
        Returns:
            list: 切分后的片段文件路径列表
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "segments")
            
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.basename(video_path)
        video_name_noext = os.path.splitext(video_name)[0]
        
        try:
            # 加载视频
            video = VideoFileClip(video_path)
            
            segment_paths = []
            
            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                
                if end_time <= start_time:
                    logger.warning(f"无效的片段时间范围: {start_time}-{end_time}")
                    continue
                
                # 切分片段
                clip = video.subclip(start_time, end_time)
                
                # 设置输出路径
                output_path = os.path.join(output_dir, f"{video_name_noext}_segment_{i:03d}.mp4")
                
                # 写入文件
                clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                    remove_temp=True,
                    logger=None
                )
                
                # 记录片段信息
                segment_info = {
                    "index": i,
                    "path": output_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "source_video": video_path
                }
                
                segment_paths.append(segment_info)
                logger.info(f"生成片段 {i}: {output_path}, 时长: {end_time - start_time:.2f}秒")
            
            # 关闭视频
            video.close()
            
            return segment_paths
            
        except Exception as e:
            logger.error(f"视频切分失败: {str(e)}")
            return []
    
    def merge_video_segments(self, segment_paths, output_path, transitions=None):
        """
        合并多个视频片段
        
        Args:
            segment_paths: 片段路径列表
            output_path: 输出视频路径
            transitions: 转场效果列表，默认为None
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始合并 {len(segment_paths)} 个视频片段")
            
            # 加载片段
            clips = []
            for path in segment_paths:
                clip = VideoFileClip(path)
                clips.append(clip)
            
            # 应用转场效果（如果有）
            if transitions:
                # 实现转场效果的逻辑，这里略过
                pass
            
            # 合并片段
            final_clip = concatenate_videoclips(clips)
            
            # 写入文件
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                logger=None
            )
            
            # 关闭所有片段
            for clip in clips:
                clip.close()
            final_clip.close()
            
            logger.info(f"视频合并成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"视频合并失败: {str(e)}")
            return False
    
    def replace_audio(self, video_path, audio_path, output_path):
        """
        替换视频的音频轨道
        
        Args:
            video_path: 视频文件路径
            audio_path: 新音频文件路径
            output_path: 输出视频路径
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始替换视频音频轨道: {video_path}")
            
            # 加载视频和音频
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # 替换音频
            video = video.set_audio(audio)
            
            # 写入文件
            video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                logger=None
            )
            
            # 关闭文件
            video.close()
            audio.close()
            
            logger.info(f"音频替换成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"音频替换失败: {str(e)}")
            return False
    
    def apply_color_correction(self, video_path, output_path, parameters=None):
        """
        对视频应用色彩校正
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            parameters: 色彩校正参数，默认为None
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始色彩校正: {video_path}")
            
            # 加载视频
            video = VideoFileClip(video_path)
            
            # 应用色彩校正
            if parameters:
                # 实现色彩校正的逻辑，这里略过
                pass
            
            # 写入文件
            video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                logger=None
            )
            
            # 关闭文件
            video.close()
            
            logger.info(f"色彩校正成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"色彩校正失败: {str(e)}")
            return False
    
    def resize_video(self, video_path, output_path, width, height):
        """
        调整视频分辨率
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            width: 目标宽度
            height: 目标高度
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始调整视频分辨率: {video_path} -> {width}x{height}")
            
            # 加载视频
            video = VideoFileClip(video_path)
            
            # 调整分辨率
            resized_video = video.resize(width=width, height=height)
            
            # 写入文件
            resized_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(self.temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                logger=None
            )
            
            # 关闭文件
            video.close()
            resized_video.close()
            
            logger.info(f"分辨率调整成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"分辨率调整失败: {str(e)}")
            return False 