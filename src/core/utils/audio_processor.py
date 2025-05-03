import os
import json
import numpy as np
import logging
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理工具类，封装音频分析与处理相关功能"""
    
    def __init__(self, temp_dir="data/temp/audio"):
        """
        初始化音频处理器
        
        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"初始化音频处理器，临时目录: {temp_dir}")
    
    def load_audio(self, audio_path):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            tuple: (音频数据, 采样率)
        """
        try:
            logger.info(f"加载音频文件: {audio_path}")
            y, sr = librosa.load(audio_path, sr=None)
            logger.info(f"音频加载成功: {audio_path}, 采样率: {sr}Hz, 时长: {len(y)/sr:.2f}秒")
            return y, sr
        except Exception as e:
            logger.error(f"音频加载失败: {str(e)}")
            return None, None
    
    def extract_audio_info(self, audio_path):
        """
        提取音频基本信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 音频信息，包括时长、采样率等
        """
        try:
            # 加载音频
            y, sr = self.load_audio(audio_path)
            
            if y is None:
                return None
            
            # 计算基本信息
            duration = len(y) / sr
            
            # 计算能量
            energy = np.sum(y**2) / len(y)
            
            # 计算过零率
            zero_crossings = librosa.zero_crossings(y)
            zcr = sum(zero_crossings) / len(zero_crossings)
            
            # 计算频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            
            # 构建音频信息
            audio_info = {
                "path": audio_path,
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,  # librosa默认加载为单声道
                "samples": len(y),
                "energy": float(energy),
                "zero_crossing_rate": float(zcr),
                "spectral_centroid": float(spectral_centroid)
            }
            
            logger.info(f"提取音频信息成功: {audio_path}")
            return audio_info
            
        except Exception as e:
            logger.error(f"提取音频信息失败: {str(e)}")
            return None
    
    def detect_speech_segments(self, audio_path, min_silence_len=700, silence_thresh=-40, keep_silence=400):
        """
        检测音频中的语音段落
        
        Args:
            audio_path: 音频文件路径
            min_silence_len: 最小静音长度(ms)
            silence_thresh: 静音阈值(dB)
            keep_silence: 保留的静音长度(ms)
            
        Returns:
            list: 语音段落列表，每个段落包含开始和结束时间
        """
        try:
            logger.info(f"开始检测语音段落: {audio_path}")
            
            # 使用pydub加载音频
            audio = AudioSegment.from_file(audio_path)
            
            # 检测非静音区域
            chunks = []
            
            # 使用PyDub的silence.detect_nonsilent
            # 此函数返回非静音区域的开始和结束时间（毫秒）
            from pydub.silence import detect_nonsilent
            chunks = detect_nonsilent(
                audio, 
                min_silence_len=min_silence_len, 
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            
            # 转换为秒为单位的段落
            segments = []
            for i, (start, end) in enumerate(chunks):
                segment = {
                    "index": i,
                    "start": start / 1000.0,  # 转换为秒
                    "end": end / 1000.0,      # 转换为秒
                    "duration": (end - start) / 1000.0
                }
                segments.append(segment)
            
            logger.info(f"检测到 {len(segments)} 个语音段落")
            return segments
            
        except Exception as e:
            logger.error(f"语音段落检测失败: {str(e)}")
            return []
    
    def detect_audio_rhythm(self, audio_path, n_fft=2048, hop_length=512):
        """
        检测音频的节奏点
        
        Args:
            audio_path: 音频文件路径
            n_fft: FFT窗口大小
            hop_length: 帧移
            
        Returns:
            list: 节奏点列表，每个点包含时间和强度
        """
        try:
            logger.info(f"开始检测音频节奏点: {audio_path}")
            
            # 加载音频
            y, sr = self.load_audio(audio_path)
            
            if y is None:
                return []
            
            # 计算音频的onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # 检测节奏点
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            
            # 转换为时间
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            
            # 计算节奏点强度
            rhythm_points = []
            for i, (frame, time) in enumerate(zip(onset_frames, onset_times)):
                # 获取该帧的强度
                if frame < len(onset_env):
                    strength = float(onset_env[frame])
                else:
                    strength = 0.0
                    
                rhythm_point = {
                    "index": i,
                    "time": float(time),
                    "strength": strength
                }
                rhythm_points.append(rhythm_point)
            
            logger.info(f"检测到 {len(rhythm_points)} 个节奏点")
            return rhythm_points
            
        except Exception as e:
            logger.error(f"节奏点检测失败: {str(e)}")
            return []
    
    def split_audio_segments(self, audio_path, segments, output_dir=None):
        """
        将音频按照指定的时间段切分成片段
        
        Args:
            audio_path: 音频文件路径
            segments: 片段列表，每个片段是一个包含开始和结束时间的字典
            output_dir: 输出目录，默认为None则使用临时目录
            
        Returns:
            list: 切分后的片段文件路径列表
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "segments")
            
        os.makedirs(output_dir, exist_ok=True)
        
        audio_name = os.path.basename(audio_path)
        audio_name_noext = os.path.splitext(audio_name)[0]
        
        try:
            # 使用pydub加载音频
            audio = AudioSegment.from_file(audio_path)
            
            segment_paths = []
            
            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0) * 1000  # 转换为毫秒
                end_time = segment.get("end", 0) * 1000      # 转换为毫秒
                
                if end_time <= start_time:
                    logger.warning(f"无效的片段时间范围: {start_time/1000}-{end_time/1000}")
                    continue
                
                # 切分片段
                audio_segment = audio[start_time:end_time]
                
                # 设置输出路径
                output_path = os.path.join(output_dir, f"{audio_name_noext}_segment_{i:03d}.wav")
                
                # 导出音频片段
                audio_segment.export(output_path, format="wav")
                
                # 记录片段信息
                segment_info = {
                    "index": i,
                    "path": output_path,
                    "start_time": start_time / 1000,  # 转换回秒
                    "end_time": end_time / 1000,      # 转换回秒
                    "duration": (end_time - start_time) / 1000,
                    "source_audio": audio_path
                }
                
                segment_paths.append(segment_info)
                logger.info(f"生成音频片段 {i}: {output_path}, 时长: {(end_time - start_time) / 1000:.2f}秒")
            
            return segment_paths
            
        except Exception as e:
            logger.error(f"音频切分失败: {str(e)}")
            return []
    
    def analyze_audio_segments(self, audio_path, transcript):
        """
        根据转录结果分析音频段落
        
        Args:
            audio_path: 音频文件路径
            transcript: 转录结果JSON对象
            
        Returns:
            list: 语音段落列表，包含文本内容和时间信息
        """
        try:
            logger.info(f"开始分析音频段落: {audio_path}")
            
            # 提取转录文本中的段落
            segments = transcript.get("segments", [])
            
            # 转换为音频段落
            audio_segments = []
            
            for i, segment in enumerate(segments):
                # 提取段落信息
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", 0)
                text = segment.get("text", "")
                
                # 跳过空文本
                if not text.strip():
                    continue
                
                # 构建段落信息
                audio_segment = {
                    "index": i,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "text": text
                }
                
                audio_segments.append(audio_segment)
            
            logger.info(f"分析了 {len(audio_segments)} 个音频段落")
            return audio_segments
            
        except Exception as e:
            logger.error(f"音频段落分析失败: {str(e)}")
            return []
    
    def extract_audio_features(self, audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            n_mfcc: MFCC特征数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            
        Returns:
            dict: 音频特征
        """
        try:
            logger.info(f"开始提取音频特征: {audio_path}")
            
            # 加载音频
            y, sr = self.load_audio(audio_path)
            
            if y is None:
                return None
            
            # 提取特征
            # 1. MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc_mean = np.mean(mfcc, axis=1).tolist()
            
            # 2. 色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            chroma_mean = np.mean(chroma, axis=1).tolist()
            
            # 3. 频谱对比度
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            contrast_mean = np.mean(contrast, axis=1).tolist()
            
            # 4. 音高
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            # 选择每帧最强音高
            pitch_mean = []
            for i in range(magnitudes.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch_mean.append(pitches[index, i])
            pitch_mean = np.mean(pitch_mean).item()
            
            # 构建特征字典
            features = {
                "path": audio_path,
                "mfcc": mfcc_mean,
                "chroma": chroma_mean,
                "contrast": contrast_mean,
                "pitch_mean": pitch_mean,
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0])
            }
            
            logger.info(f"音频特征提取成功: {audio_path}")
            return features
            
        except Exception as e:
            logger.error(f"音频特征提取失败: {str(e)}")
            return None
    
    def save_audio_analysis(self, analysis, output_path):
        """
        保存音频分析结果到文件
        
        Args:
            analysis: 音频分析结果
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"音频分析结果已保存到 {output_path}")
        except Exception as e:
            logger.error(f"保存音频分析结果失败: {str(e)}")
    
    def time_stretch(self, audio_path, output_path, speed_factor):
        """
        时间拉伸音频（变速不变调）
        
        Args:
            audio_path: 音频文件路径
            output_path: 输出文件路径
            speed_factor: 速度因子，>1加速，<1减速
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始时间拉伸音频: {audio_path}, 速度因子: {speed_factor}")
            
            # 加载音频
            y, sr = self.load_audio(audio_path)
            
            if y is None:
                return False
            
            # 时间拉伸
            y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
            
            # 保存音频
            sf.write(output_path, y_stretched, sr)
            
            logger.info(f"时间拉伸成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"时间拉伸失败: {str(e)}")
            return False 