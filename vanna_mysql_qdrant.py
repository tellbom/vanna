from vanna.openai import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Union, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import json
from elasticsearch import Elasticsearch
import re
from functools import wraps
import logging
# 从vanna.qdrant导入Qdrant_VectorStore而不是ChromaDB相关类
from vanna.qdrant import Qdrant_VectorStore

from flask import jsonify, request
import time
import logging
import traceback
from typing import Dict, Any, List, Tuple

import logging
import time
import requests
from typing import List, Dict, Any, Optional

from typing import Any, List, Dict
import json

import threading
import queue
import time
import uuid
from typing import Dict, Callable, Any, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVanna")

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="sk-9f8124e18aa242af830c8a502c015c40",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


class AsyncTaskManager:
    """异步任务管理器，使用线程池处理耗时任务"""

    def __init__(self, max_workers=5):
        """
        初始化任务管理器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.workers = []
        self.results = {}  # 存储任务结果
        self.status = {}  # 存储任务状态 (pending, running, completed, failed)
        self.callbacks = {}  # 任务完成后的回调函数
        self.logger = logging.getLogger("AsyncTaskManager")
        self._start_workers()

    def _worker_loop(self):
        """工作线程循环，不断从队列获取任务并执行"""
        while True:
            try:
                # 从队列获取任务
                task_id, task_func, args, kwargs = self.task_queue.get()

                # 更新任务状态
                self.status[task_id] = "running"
                self.logger.info(f"开始执行任务 {task_id}")

                try:
                    # 执行任务
                    result = task_func(*args, **kwargs)
                    # 存储结果
                    self.results[task_id] = result
                    self.status[task_id] = "completed"
                    self.logger.info(f"任务 {task_id} 完成")

                    # 执行回调（如果有）
                    if task_id in self.callbacks and self.callbacks[task_id]:
                        try:
                            self.callbacks[task_id](result)
                            self.logger.info(f"任务 {task_id} 回调执行成功")
                        except Exception as e:
                            self.logger.error(f"任务 {task_id} 回调执行失败: {str(e)}")

                except Exception as e:
                    # 任务执行失败
                    self.status[task_id] = "failed"
                    self.results[task_id] = str(e)
                    self.logger.error(f"任务 {task_id} 执行失败: {str(e)}")

                # 标记任务完成
                self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"工作线程执行出错: {str(e)}")
                # 短暂休息以避免CPU占用过高
                time.sleep(0.1)

    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.workers.append(thread)
            self.logger.info(f"启动工作线程 {i + 1}")

    def submit_task(self, task_func: Callable, callback: Optional[Callable] = None, *args, **kwargs) -> str:
        """
        提交任务到队列

        Args:
            task_func: 要执行的函数
            callback: 任务完成后的回调函数，接收任务结果作为参数
            *args, **kwargs: 传递给任务函数的参数

        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        self.status[task_id] = "pending"

        if callback:
            self.callbacks[task_id] = callback

        # 将任务放入队列
        self.task_queue.put((task_id, task_func, args, kwargs))
        self.logger.info(f"提交任务 {task_id} 到队列")

        return task_id

    def get_task_status(self, task_id: str) -> Dict:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict: 包含任务状态和结果（如果已完成）
        """
        if task_id not in self.status:
            return {"status": "not_found"}

        result = {
            "status": self.status[task_id]
        }

        # 如果任务已完成或失败，包含结果
        if self.status[task_id] in ["completed", "failed"] and task_id in self.results:
            result["result"] = self.results[task_id]

        return result

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        等待任务完成并返回结果

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            Any: 任务结果

        Raises:
            TimeoutError: 如果等待超时
            ValueError: 如果任务不存在
            RuntimeError: 如果任务执行失败
        """
        if task_id not in self.status:
            raise ValueError(f"任务 {task_id} 不存在")

        start_time = time.time()
        while self.status[task_id] in ["pending", "running"]:
            time.sleep(0.1)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务 {task_id} 超时")

        if self.status[task_id] == "failed":
            raise RuntimeError(f"任务 {task_id} 执行失败: {self.results[task_id]}")

        return self.results[task_id]

    def clean_old_tasks(self, max_age: float = 3600):
        """
        清理旧任务数据

        Args:
            max_age: 最大保留时间（秒），默认1小时
        """
        # 实现清理逻辑...
        pass


class DocumentChunker:
    def __init__(self, mechanical_terms_path="/dictionary/MechanicalWords.txt", max_chunk_size=1000, overlap=50,
                 model_name="m3e-base"):
        """
        初始化文档分块器，使用语义模型辅助优化分块

        Args:
            mechanical_terms_path: 专业术语词典文件路径
            max_chunk_size: 最大块大小
            overlap: 块之间的重叠大小
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.terms_dict = self._load_mechanical_terms(mechanical_terms_path)

        # 初始化语义模型
        self.semantic_model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

    def _load_mechanical_terms(self, filepath):
        """加载专业术语词典 - 调整为适应一行英文一行中文的格式"""
        terms_dict = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines) - 1:
                    en_term = lines[i].strip()
                    zh_term = lines[i + 1].strip()
                    if en_term and zh_term:
                        terms_dict[en_term] = zh_term
                    i += 2
        except Exception as e:
            logger.warning(f"加载专业术语词典失败: {str(e)}")

        return terms_dict

    def _protect_terms(self, text):
        """保护专业术语不被分割"""
        protected_text = text
        for en_term, zh_term in self.terms_dict.items():
            if en_term in text:
                protected_text = protected_text.replace(en_term, f"[TERM_{en_term}_TERM]")
            if zh_term in text:
                protected_text = protected_text.replace(zh_term, f"[TERM_{zh_term}_TERM]")
        return protected_text

    def _restore_terms(self, text):
        """恢复专业术语"""
        restored_text = text
        for en_term, zh_term in self.terms_dict.items():
            restored_text = restored_text.replace(f"[TERM_{en_term}_TERM]", en_term)
            restored_text = restored_text.replace(f"[TERM_{zh_term}_TERM]", zh_term)
        return restored_text

    def is_semantic_boundary(self, left_text, right_text, threshold=0.6):
        """
        使用语义模型判断是否为合适的分块边界

        Args:
            left_text: 左侧文本
            right_text: 右侧文本
            threshold: 相似度阈值，低于此值视为合适的边界

        Returns:
            bool: 是否为合适的分块边界
        """
        # 边界检查：确保文本非空
        if not left_text or not right_text:
            return True

        # 取有限的上下文窗口
        left_context = left_text[-150:] if len(left_text) > 150 else left_text
        right_context = right_text[:150] if len(right_text) > 150 else right_text

        # 生成嵌入
        left_emb = self.semantic_model.encode(left_context, convert_to_numpy=True)
        right_emb = self.semantic_model.encode(right_context, convert_to_numpy=True)

        # 计算余弦相似度
        similarity = np.dot(left_emb, right_emb) / (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))

        # 相似度低于阈值，认为是合适的分块边界
        return similarity < threshold

    def chunk_document(self, document):
        """
        主分块方法，结合固定格式特征和语义边界检测，优化处理大型表格
        """
        # 边界检查：空文档处理
        if not document or not document.strip():
            return []

        # 边界检查：如果文档小于最大块大小，直接返回
        if len(document) <= self.max_chunk_size:
            return [document]

        # 保护专业术语
        protected_text = self._protect_terms(document)

        # 1. 首先尝试按表定义分块
        table_sections = []
        table_pattern = r'(# 表名词:.+?)(?=# 表名词:|$)'
        table_matches = re.finditer(table_pattern, protected_text, re.DOTALL)

        for match in table_matches:
            table_sections.append(match.group(1))

        # 如果按表分块成功，优先处理表级别分块
        if table_sections:
            result_chunks = []

            for section in table_sections:
                # 检测是否包含表格
                table_start = section.find("\n|")

                if table_start > 0:
                    # 检查表格大小
                    lines = section.split("\n")
                    header_line_idx = -1
                    separator_line_idx = -1

                    # 查找表头行和分隔行
                    for i, line in enumerate(lines):
                        if line.strip().startswith("|") and header_line_idx == -1:
                            header_line_idx = i
                        elif line.strip().startswith("|") and header_line_idx != -1 and "-" in line:
                            # 分隔行通常含有连字符
                            separator_line_idx = i
                            break

                    # 计算列数
                    if header_line_idx != -1:
                        header_line = lines[header_line_idx]
                        column_count = header_line.count("|") - 1

                        # 大表格处理策略(超过20列视为大表格)
                        if column_count > 20:
                            # 提取表前内容
                            before_table = "\n".join(lines[:header_line_idx])
                            if before_table:
                                result_chunks.append(before_table)

                            # 获取表头部分
                            table_header = "\n".join(
                                lines[header_line_idx:separator_line_idx + 1]) if separator_line_idx != -1 else lines[
                                header_line_idx]

                            # 计算表格主体范围
                            table_body_start = separator_line_idx + 1 if separator_line_idx != -1 else header_line_idx + 1
                            table_body_end = table_body_start

                            # 查找表格结束位置
                            for i in range(table_body_start, len(lines)):
                                if lines[i].strip().startswith("|"):
                                    table_body_end = i
                                elif table_body_end >= table_body_start and lines[i].strip() and not lines[
                                    i].strip().startswith("|"):
                                    break

                            # 获取表格数据行
                            data_rows = lines[table_body_start:table_body_end + 1]

                            # 表格分块：每个块含表头+少量数据行
                            header_context = f"## 示例数据\n\n{table_header}\n"
                            max_rows_per_chunk = 5  # 每块最多5行数据

                            for i in range(0, len(data_rows), max_rows_per_chunk):
                                chunk_rows = data_rows[i:i + max_rows_per_chunk]
                                table_chunk = header_context + "\n".join(chunk_rows)
                                result_chunks.append(table_chunk)

                            # 处理表格后内容
                            if table_body_end + 1 < len(lines):
                                after_table = "\n".join(lines[table_body_end + 1:])
                                if len(after_table) <= self.max_chunk_size:
                                    result_chunks.append(after_table)
                                else:
                                    # 继续使用原有的分块方法处理剩余内容
                                    after_chunks = self._process_section_with_original_method(after_table)
                                    result_chunks.extend(after_chunks)
                        else:
                            # 小表格处理，尝试保持完整
                            if len(section) <= self.max_chunk_size:
                                result_chunks.append(section)
                            else:
                                # 回退到原有分块逻辑
                                section_chunks = self._process_section_with_original_method(section)
                                result_chunks.extend(section_chunks)
                    else:
                        # 没有找到表头，使用原有分块逻辑
                        section_chunks = self._process_section_with_original_method(section)
                        result_chunks.extend(section_chunks)
                else:
                    # 没有表格，使用原有分块逻辑
                    section_chunks = self._process_section_with_original_method(section)
                    result_chunks.extend(section_chunks)

            # 恢复专业术语并返回
            return [self._restore_terms(chunk) for chunk in result_chunks]

        # 2. 如果不是按表结构组织的，回退到原有的基于标记的分块方法
        # 找出所有关键标记的位置
        key_markers = [
            "业务场景:",
            "示例数据:",
            "业务规则:"
        ]

        marker_positions = []
        for marker in key_markers:
            for match in re.finditer(re.escape(marker), protected_text):
                marker_positions.append(match.start())

        # 加入文本开始和结束位置
        marker_positions = [0] + sorted(marker_positions) + [len(protected_text)]

        # 生成初步分块
        initial_chunks = []
        for i in range(len(marker_positions) - 1):
            start = marker_positions[i]
            end = marker_positions[i + 1]

            chunk_text = protected_text[start:end]
            if chunk_text.strip():  # 忽略空块
                initial_chunks.append(chunk_text)

        # 3. 对超出最大长度的块进行进一步分割
        result_chunks = []
        for chunk in initial_chunks:
            if len(chunk) <= self.max_chunk_size:
                result_chunks.append(chunk)
            else:
                # 按段落分割
                paragraphs = re.split(r'\n\s*\n', chunk)
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n"
                        current_chunk += para
                    else:
                        # 当前段落会导致超出最大长度
                        if current_chunk:
                            # 检查语义边界
                            window_size = min(100, len(current_chunk), len(para))
                            if self.is_semantic_boundary(
                                    current_chunk[-window_size:],
                                    para[:window_size]
                            ):
                                # 是语义边界，直接分块
                                result_chunks.append(current_chunk)
                                current_chunk = para
                            else:
                                # 不是理想的语义边界，但必须分块
                                result_chunks.append(current_chunk)
                                current_chunk = para
                        else:
                            # 单个段落太长，需要强制分割
                            if len(para) > self.max_chunk_size:
                                sub_chunks = self._split_long_paragraph(para)
                                result_chunks.extend(sub_chunks)
                            else:
                                current_chunk = para

                if current_chunk:
                    result_chunks.append(current_chunk)

        # 4. 确保相邻块有适当的重叠
        final_chunks = []
        for i, chunk in enumerate(result_chunks):
            if i > 0 and self.overlap > 0:
                # 添加上一个块的末尾作为重叠
                prev_end = result_chunks[i - 1][-self.overlap:] if len(result_chunks[i - 1]) > self.overlap else \
                    result_chunks[i - 1]
                if not chunk.startswith(prev_end):
                    chunk = prev_end + chunk
            final_chunks.append(chunk)

        # 5. 恢复专业术语并返回
        return [self._restore_terms(chunk) for chunk in final_chunks]

    def _process_section_with_original_method(self, section):
        """使用原有分块方法处理部分内容"""
        if len(section) <= self.max_chunk_size:
            return [section]

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', section)
        current_chunk = ""
        result_chunks = []

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                # 检查语义边界
                if current_chunk:
                    window_size = min(100, len(current_chunk), len(para))
                    if self.is_semantic_boundary(
                            current_chunk[-window_size:],
                            para[:window_size]
                    ):
                        # 是语义边界，直接分块
                        result_chunks.append(current_chunk)
                        current_chunk = para
                    else:
                        # 不是理想的语义边界，但必须分块
                        result_chunks.append(current_chunk)
                        current_chunk = para
                else:
                    # 单个段落太长，需要强制分割
                    if len(para) > self.max_chunk_size:
                        sub_chunks = self._split_long_paragraph(para)
                        result_chunks.extend(sub_chunks)
                    else:
                        current_chunk = para

        if current_chunk:
            result_chunks.append(current_chunk)

        return result_chunks

    def _split_long_paragraph(self, paragraph):
        """处理超长段落，使用语义边界判断"""
        # 边界检查
        if len(paragraph) <= self.max_chunk_size:
            return [paragraph]

        chunks = []
        remaining = paragraph

        # 防止无限循环
        safety_counter = 0
        max_iterations = 100

        while len(remaining) > self.max_chunk_size and safety_counter < max_iterations:
            safety_counter += 1

            # 找一个接近最大长度的句子边界
            potential_end = self.max_chunk_size

            # 尝试在附近找句号等标点
            sentence_endings = [m.start() for m in re.finditer(r'[。！？!?]', remaining[:self.max_chunk_size + 100])]
            if sentence_endings:
                closest_ending = min(sentence_endings, key=lambda x: abs(x - self.max_chunk_size))
                if closest_ending > 0:
                    potential_end = closest_ending + 1  # 包括标点

            # 检查语义边界
            if potential_end < len(remaining) - 50:
                left = remaining[:potential_end]
                right = remaining[potential_end:potential_end + 100]

                if self.is_semantic_boundary(left, right):
                    chunks.append(remaining[:potential_end])
                    remaining = remaining[potential_end:]
                else:
                    # 不理想但必须分割
                    chunks.append(remaining[:potential_end])
                    remaining = remaining[potential_end:]
            else:
                # 剩余部分不长，全部加入
                chunks.append(remaining)
                break

        # 添加剩余部分
        if remaining and safety_counter < max_iterations:
            chunks.append(remaining)

        return chunks



# 修正utils类中的serialize_data函数
class utils:
    @staticmethod
    def serialize_data(data):
        """序列化数据，确保JSON兼容"""
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        if isinstance(data, list):
            return [utils.serialize_data(item) for item in data]

        if isinstance(data, dict):
            return {str(k): utils.serialize_data(v) for k, v in data.items()}

        # 尝试转换为字符串
        try:
            return str(data)
        except:
            return "不可序列化的值"


class RetrievalService:
    """封装检索和重排序逻辑的服务类"""

    def __init__(self, vanna_instance, config=None):
        """
        初始化检索服务

        Args:
            vanna_instance: Vanna实例，用于执行检索操作
            config: 配置字典
        """
        self.vn = vanna_instance
        self.config = config or {}

        # 配置重排序服务
        self.rerank_url = self.config.get("rerank_url", "http://localhost:8091")
        self.rerank_enabled = self.config.get("rerank_enabled", True)
        self.rerank_timeout = self.config.get("rerank_timeout", 120)

        # 检索配置
        self.max_results = self.config.get("max_results", 10)

        logger.info(f"检索服务初始化完成, 重排序{'启用' if self.rerank_enabled else '禁用'}")

    def retrieve(self, question: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute comprehensive retrieval pipeline using the new integrated reranking approach

        Args:
            question: User question
            options: Retrieval options including:
                - max_results: Maximum results per type
                - enhance_query: Whether to enhance the query
                - use_rerank: Whether to use reranking
                - rerank_top_k: Number of candidates to rerank

        Returns:
            Dict containing retrieval results and statistics
        """
        start_time = time.time()
        options = options or {}
        stats = {}

        # Extract options
        max_results = options.get("max_results", self.max_results)
        enhance_query = options.get("enhance_query", True)
        use_rerank = options.get("use_rerank", self.rerank_enabled)
        rerank_top_k = options.get("rerank_top_k", max_results * 2)  # Default to twice max_results

        # Prepare retrieval kwargs
        retrieval_kwargs = {
            'rerank': use_rerank,
            'rerank_top_k': rerank_top_k
        }

        # Step 1: Enhance query if needed (handled internally by each method now)
        if enhance_query and hasattr(self.vn, "preprocess_field_names"):
            enhanced_question = self.vn.preprocess_field_names(question)
            stats["enhanced_question"] = enhanced_question

        # Step 2: Retrieve SQL examples with integrated reranking
        sql_start = time.time()
        question_sql_list = self.vn.get_similar_question_sql(question, **retrieval_kwargs)
        stats["sql_retrieval_time"] = round(time.time() - sql_start, 3)
        stats["sql_results_count"] = len(question_sql_list)

        # Step 3: Retrieve DDL with integrated reranking
        ddl_start = time.time()
        ddl_list = self.vn.get_related_ddl(question, **retrieval_kwargs)
        stats["ddl_retrieval_time"] = round(time.time() - ddl_start, 3)
        stats["ddl_results_count"] = len(ddl_list)

        # Step 4: Retrieve docs with integrated reranking
        doc_start = time.time()
        doc_list = self.vn.get_related_documentation(question, **retrieval_kwargs)
        stats["doc_retrieval_time"] = round(time.time() - doc_start, 3)
        stats["doc_results_count"] = len(doc_list)

        # Step 5: Extract table relationships
        table_relationships = ""
        if hasattr(self.vn, "_extract_table_relationships"):
            table_relationships = self.vn._extract_table_relationships(ddl_list)

        # Step 6: Limit results if needed
        if max_results > 0:
            if len(question_sql_list) > max_results:
                question_sql_list = question_sql_list[:max_results]
            if len(ddl_list) > max_results:
                ddl_list = ddl_list[:max_results]
            if len(doc_list) > max_results:
                doc_list = doc_list[:max_results]

        # Return the results
        results = {
            "question_sql_list": question_sql_list,
            "ddl_list": ddl_list,
            "doc_list": doc_list,
            "table_relationships": table_relationships,
            "stats": stats,
            "total_time": round(time.time() - start_time, 3)
        }

        return results

# 向量和BM25结果融合算法
class RankFusion:
    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[Any]], weights: List[float] = None, k: int = 60) -> List[Any]:
        """
        使用加权互惠排序融合（RRF）将多个结果列表融合。

        Args:
            results_lists: 结果列表的列表，每个列表按相关性排序
            weights: 各个结果列表的权重，默认权重相等
            k: RRF常数，控制排名的影响

        Returns:
            融合后的结果列表
        """
        if weights is None:
            weights = [1.0] * len(results_lists)

        if len(weights) != len(results_lists):
            raise ValueError("权重列表长度必须与结果列表数量相同")

        scores = {}

        # 为每个结果列表计算RRF分数
        for list_idx, (result_list, weight) in enumerate(zip(results_lists, weights)):
            for rank, item in enumerate(result_list, start=1):
                # 对JSON对象，转为字符串作为键
                if isinstance(item, dict):
                    item_key = json.dumps(item, sort_keys=True)
                else:
                    item_key = str(item)

                # 初始化该项的记录
                if item_key not in scores:
                    scores[item_key] = {"item": item, "score": 0.0, "sources": set()}

                # 计算加权RRF分数并累加
                scores[item_key]["score"] += weight * (1.0 / (k + rank))
                scores[item_key]["sources"].add(list_idx)

        # 上下文奖励：出现在多个来源的结果获得额外奖励
        for item_key, data in scores.items():
            # 来源多样性奖励 - 同时出现在多个搜索结果中的项目得分提升
            source_diversity = len(data["sources"]) / len(results_lists)
            data["score"] *= (1 + 0.2 * source_diversity)  # 最多20%的奖励

        # 按分数降序排序
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

    @staticmethod
    def multi_stage_fusion(query: str, search_functions: Dict[str, Tuple[callable, float]], k: int = 60) -> List[Any]:
        """
        多阶段融合算法，针对相近语义字段进行优化

        Args:
            query: 原始查询
            search_functions: 搜索函数字典，格式为 {名称: (搜索函数, 权重)}
            k: RRF常数

        Returns:
            融合后的结果列表
        """
        # 第一阶段: 直接检索
        stage1_results = {}
        stage1_lists = []
        stage1_weights = []

        for name, (search_fn, weight) in search_functions.items():
            results = search_fn(query)
            stage1_results[name] = results
            stage1_lists.append(results)
            stage1_weights.append(weight)

        # 第二阶段: 提取关键词
        key_terms = RankFusion._extract_key_terms(query, stage1_results)

        # 第三阶段: 使用关键词增强查询进行二次检索
        stage2_results = {}
        for term in key_terms:
            enhanced_query = f"{query} {term}"
            term_results = []

            for name, (search_fn, _) in search_functions.items():
                results = search_fn(enhanced_query)
                term_results.extend(results)

            stage2_results[term] = term_results

        # 第四阶段: 融合所有结果
        all_candidates = {}

        # 添加第一阶段结果
        for name, results in stage1_results.items():
            weight = search_functions[name][1]
            for rank, item in enumerate(results, start=1):
                item_key = RankFusion._get_item_key(item)

                if item_key not in all_candidates:
                    all_candidates[item_key] = {"item": item, "score": 0, "sources": set()}

                all_candidates[item_key]["score"] += weight * (1.0 / (k + rank))
                all_candidates[item_key]["sources"].add(f"direct_{name}")

        # 添加第二阶段结果 (关键词增强)
        term_weight = 0.3 / len(stage2_results) if stage2_results else 0
        for term, results in stage2_results.items():
            for rank, item in enumerate(results, start=1):
                item_key = RankFusion._get_item_key(item)

                if item_key not in all_candidates:
                    all_candidates[item_key] = {"item": item, "score": 0, "sources": set()}

                all_candidates[item_key]["score"] += term_weight * (1.0 / (k + rank))
                all_candidates[item_key]["sources"].add(f"term_{term}")

        # 多样性奖励
        for item_key, data in all_candidates.items():
            # 基于来源数量的多样性奖励
            source_count = len(data["sources"])
            direct_sources = sum(1 for s in data["sources"] if s.startswith("direct_"))
            term_sources = sum(1 for s in data["sources"] if s.startswith("term_"))

            # 混合来源的结果获得更高奖励
            if direct_sources > 0 and term_sources > 0:
                data["score"] *= 1.3  # 30%的奖励

            # 来源数量越多，奖励越高
            data["score"] *= (1 + 0.1 * min(5, source_count) / 5)

        # 按分数排序
        sorted_items = sorted(all_candidates.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

    @staticmethod
    def _extract_key_terms(query: str, results: Dict[str, List]) -> List[str]:
        """从查询和结果中提取可能的关键词"""
        terms = []

        # 1. 从查询中提取潜在字段名和值
        # 匹配可能的字段名
        field_patterns = [
            (r'([a-z]+[A-Z][a-zA-Z]*)', 1),  # 驼峰命名
            (r'([a-z_]+)', 1),  # 下划线命名
            (r'"([^"]+)"', 1),  # 双引号包围
            (r"'([^']+)'", 1)  # 单引号包围
        ]

        for pattern, group in field_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m and len(m) > 2:  # 忽略太短的词
                            terms.append(m)
                elif match and len(match) > 2:
                    terms.append(match)

        # 2. 提取特殊业务术语
        business_terms = {
            "订单": ["order", "purchase"],
            "客户": ["customer", "client", "user"],
            "产品": ["product", "item", "goods"],
            "支付": ["payment", "transaction"],
            "状态": ["status", "state", "condition"],
            "取消": ["cancel", "cancelled", "canceled", "cancellation"],
            "完成": ["complete", "completed", "finish", "finished"],
            "收货": ["delivery", "delivered", "receive", "received"]
        }

        for term, synonyms in business_terms.items():
            if term in query:
                terms.extend(synonyms)

        # 3. 转换驼峰/下划线变体
        variants = []
        for term in terms:
            # 驼峰转下划线
            if any(c.isupper() for c in term):
                snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in term]).lstrip('_')
                variants.append(snake_case)

            # 下划线转驼峰
            if '_' in term:
                camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(term.split('_')))
                variants.append(camel_case)

        terms.extend(variants)

        # 4. 从结果中提取高频词
        # 这里可以实现更复杂的逻辑

        # 去重并返回
        return list(set(terms))

    @staticmethod
    def _get_item_key(item) -> str:
        """获取项目的唯一键"""
        if isinstance(item, dict):
            return json.dumps(item, sort_keys=True)
        return str(item)

    @staticmethod
    def contextual_fusion(query: str, dense_results: list, lexical_results: list, metadata=None, k: int = 60,
                          rerank_service_url: str = None) -> list:
        """
        Context-aware fusion algorithm with integrated reranking for semantic field optimization

        Args:
            query: User query
            dense_results: Vector search results (or reranked results)
            lexical_results: Text search results
            metadata: Metadata information
            k: RRF constant
            rerank_service_url: Optional URL for reranking service

        Returns:
            Fused result list
        """
        # Extract query features
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        is_status_query = any(
            term in query_terms for term in ['状态', '取消', '完成', '支付', 'status', 'complete', 'cancel'])
        is_time_query = any(
            term in query_terms for term in ['时间', '日期', '年', '月', '日', 'time', 'date', 'year', 'month', 'day'])
        is_type_query = any(term in query_terms for term in ['类型', '种类', '分类', 'type', 'category', 'kind'])
        is_count_query = any(term in query_terms for term in ['数量', '多少', '计数', 'count', 'how many', 'number of'])

        # Query intent detection - expand to more intents based on query structure
        is_aggregation_query = any(term in query.lower() for term in
                                   ['average', 'sum', 'total', 'count', 'mean', 'max', 'min', '平均', '总和', '最大',
                                    '最小'])
        is_comparison_query = any(term in query.lower() for term in
                                  ['more than', 'less than', 'greater', 'smaller', 'between', 'compare', '超过', '小于',
                                   '大于', '比较'])

        # Set dynamic weights based on query intent
        if is_status_query or is_type_query:
            # Status and type queries, BM25 might be more accurate
            vector_weight = 0.4
            lexical_weight = 0.6
        elif is_time_query:
            # Time queries, both are important
            vector_weight = 0.5
            lexical_weight = 0.5
        elif is_count_query or is_aggregation_query:
            # Count/aggregation queries often need precise lexical matching
            vector_weight = 0.35
            lexical_weight = 0.65
        elif is_comparison_query:
            # Comparison queries benefit from semantic understanding
            vector_weight = 0.6
            lexical_weight = 0.4
        else:
            # Default weights with preference for vector/semantic results
            vector_weight = 0.7
            lexical_weight = 0.3

        # Apply external reranking if URL is provided (as a final step after fusion)
        if rerank_service_url:
            try:
                # Combine all candidates for reranking
                all_candidates = []
                seen_items = set()

                # Process dense results first (higher priority)
                for item in dense_results:
                    item_key = RankFusion._get_item_key(item)
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        all_candidates.append({"content": str(item), "item": item})

                # Add lexical results
                for item in lexical_results:
                    item_key = RankFusion._get_item_key(item)
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        all_candidates.append({"content": str(item), "item": item})

                # Call reranking service
                response = requests.post(
                    rerank_service_url,
                    json={
                        "query": query,
                        "documents": [{"content": c["content"]} for c in all_candidates],
                        "top_k": len(all_candidates)
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    # Process reranked results
                    reranked_data = response.json()
                    reranked_results = []

                    # Reconstruct original items in new order
                    for item in reranked_data.get("results", []):
                        idx = item.get("index")
                        if 0 <= idx < len(all_candidates):
                            reranked_results.append(all_candidates[idx]["item"])

                    # Return reranked results directly
                    return reranked_results

            except Exception as e:
                # Log error and continue with regular fusion
                print(f"Error during reranking in contextual_fusion: {str(e)}")

        # Compute fusion scores using RRF as base
        scores = {}

        # Process vector results
        for rank, item in enumerate(dense_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            # Check if item has a rerank_score from previous reranking
            if isinstance(item, dict) and "rerank_score" in item:
                # Use rerank score as a boost
                rerank_boost = min(1.5, 1.0 + item["rerank_score"] / 2)  # Cap at 50% boost
                scores[item_key]["score"] += vector_weight * (1.0 / (k + rank)) * rerank_boost
            else:
                scores[item_key]["score"] += vector_weight * (1.0 / (k + rank))

            scores[item_key]["matches"].add("vector")

        # Process lexical results
        for rank, item in enumerate(lexical_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            scores[item_key]["score"] += lexical_weight * (1.0 / (k + rank))
            scores[item_key]["matches"].add("lexical")

            # Calculate exact term match bonus
            item_str = str(item).lower()

            # Check if result contains query terms
            term_matches = sum(1 for term in query_terms if term in item_str)
            term_match_ratio = term_matches / len(query_terms) if query_terms else 0

            # Term matching bonus increases with match ratio
            term_match_boost = 1.0 + (0.3 * term_match_ratio)
            scores[item_key]["score"] *= term_match_boost

        # Multiple retrieval source bonus
        for item_key, data in scores.items():
            if len(data["matches"]) > 1:
                data["score"] *= 1.25  # 25% extra score for items appearing in both retrieval methods

        # Add intent-specific bonuses based on pattern matching
        for item_key, data in scores.items():
            item_str = str(data["item"]).lower()

            # Status query bonuses
            if is_status_query and any(
                    term in item_str for term in ["status", "state", "condition", "状态", "完成", "取消", "进行中"]):
                data["score"] *= 1.15

            # Time query bonuses
            if is_time_query and any(
                    term in item_str for term in ["time", "date", "timestamp", "时间", "日期", "年", "月", "日"]):
                data["score"] *= 1.15

            # Type query bonuses
            if is_type_query and any(term in item_str for term in ["type", "category", "类型", "种类", "分类"]):
                data["score"] *= 1.15

            # Count/aggregation query bonuses
            if (is_count_query or is_aggregation_query) and any(
                    term in item_str for term in ["count", "sum", "avg", "average", "total", "数量", "总数", "平均"]):
                data["score"] *= 1.18

            # Comparison query bonuses
            if is_comparison_query and any(term in item_str for term in
                                           ["greater", "less", "between", "compare", "大于", "小于", "之间", "比较"]):
                data["score"] *= 1.15

        # Sort by score
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item_data["item"] for item_data in sorted_items]

# 本地 Sentence Transformer 嵌入函数
class LocalSentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        # 加载本地模型，禁止网络访问
        self.model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        # 生成嵌入向量
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


# 扩展的Vanna类 - 集成ES和Qdrant向量数据库
class EnhancedVanna(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        self.config = config or {}
        model_name = self.config.get("embedding_model", "bge-m3")

        # 创建本地 SentenceTransformer 嵌入模型实例
        self.embedding_model = SentenceTransformer(
            model_name_or_path=f"/models/sentence-transformers_{model_name}",
            local_files_only=True
        )

        # 配置Qdrant连接参数
        qdrant_config = {
            "url": self.config.get("qdrant_url", "http://localhost:6333"),
            "api_key": self.config.get("qdrant_api_key", None),
            "prefer_grpc": self.config.get("prefer_grpc", True),
            "timeout": self.config.get("qdrant_timeout", 30),
            "sql_collection_name": self.config.get("sql_collection_name", "vanna_sql"),
            "ddl_collection_name": self.config.get("ddl_collection_name", "vanna_ddl"),
            "documentation_collection_name": self.config.get("documentation_collection_name", "vanna_documentation"),
            "n_results": self.config.get("n_results", 10),
        }


        # 初始化Qdrant_VectorStore基类
        Qdrant_VectorStore.__init__(self, config=qdrant_config)

        # 初始化OpenAI_Chat基类
        OpenAI_Chat.__init__(self, client=client, config=self.config)

        # 初始化ES客户端（如果配置中指定）
        self.es_client = None
        self.es_indexes = {
            "sql": self.config.get("es_sql_index", "vanna_sql"),
            "ddl": self.config.get("es_ddl_index", "vanna_ddl"),
            "documentation": self.config.get("es_documentation_index", "vanna_documentation")
        }
        self.fusion_weights = {
            "vector": self.config.get("vector_weight", 0.7),
            "bm25": self.config.get("bm25_weight", 0.3)
        }
        self.fusion_method = self.config.get("fusion_method", "rrf")

        # 连接到Elasticsearch
        self._connect_elasticsearch()

        # 猴子补丁原始train方法以确保与ES同步
        self._original_train = self.train
        self.train = self._patched_train

        # 保存检索参数
        self.n_results_sql = self.config.get("n_results_sql", 10)
        self.n_results_ddl = self.config.get("n_results_ddl", 10)
        self.n_results_documentation = self.config.get("n_results_documentation", 10)

    # 重写 generate_embedding 方法，使用我们的本地模型
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        """
        使用本地 SentenceTransformer 模型生成文本嵌入向量

        Args:
            data (str): 需要生成嵌入的文本

        Returns:
            List[float]: 嵌入向量
        """
        try:
            # 使用预先初始化的 embedding_model 生成嵌入向量
            embedding = self.embedding_model.encode(data, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")

    def _connect_elasticsearch(self):
        """连接到Elasticsearch服务器"""
        es_config = self.config.get("elasticsearch", {})
        if not es_config:
            logger.info("未提供Elasticsearch配置，跳过ES初始化")
            return

        try:
            # 创建ES客户端
            self.es_client = Elasticsearch(
                hosts=es_config.get("hosts", ["http://124.71.225.73:9200"]),
                basic_auth=(
                    es_config.get("username", ""),
                    es_config.get("password", "")
                ) if es_config.get("username") else None,
                verify_certs=es_config.get("verify_certs", True),
                timeout=es_config.get("timeout", 30)
            )

            # 验证连接
            if self.es_client.ping():
                logger.info("成功连接到Elasticsearch")

                # 检查并创建索引
                self._create_es_indexes()
            else:
                logger.warning("无法连接到Elasticsearch")
                self.es_client = None
        except Exception as e:
            logger.error(f"连接Elasticsearch时出错: {str(e)}")
            self.es_client = None

    def _create_es_indexes(self):
        """确保所需的ES索引存在"""
        if not self.es_client:
            return

        # 索引映射定义
        mappings = {
            "properties": {
                "document": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "id": {
                    "type": "keyword"
                }
            }
        }

        # 检查并创建索引
        for index_name in self.es_indexes.values():
            if not self.es_client.indices.exists(index=index_name):
                self.es_client.indices.create(
                    index=index_name,
                    body={
                        "mappings": mappings,
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "ik_smart": {
                                        "type": "custom",
                                        "tokenizer": "ik_smart"
                                    },
                                    "ik_max_word": {
                                        "type": "custom",
                                        "tokenizer": "ik_max_word"
                                    }
                                }
                            }
                        }
                    }
                )
                logger.info(f"创建索引 {index_name}")

    def _index_to_es(self, collection_type: str, document: str, id: str):
        """将文档索引到ES"""
        if not self.es_client:
            return

        index_name = self.es_indexes.get(collection_type)
        if not index_name:
            return

        try:
            self.es_client.index(
                index=index_name,
                id=id,
                document={"document": document, "id": id}
            )
            logger.debug(f"文档已索引到 {index_name}, ID: {id}")
        except Exception as e:
            logger.error(f"索引文档到ES时出错: {str(e)}")

    def _search_es(self, collection_type: str, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """从ES中搜索文档"""
        if not self.es_client:
            return []

        index_name = self.es_indexes.get(collection_type)
        if not index_name:
            return []

        try:
            # 使用BM25进行全文搜索
            response = self.es_client.search(
                index=index_name,
                body={
                    "query": {
                        "match": {
                            "document": query
                        }
                    },
                    "size": size
                }
            )

            # 提取结果
            hits = response.get("hits", {}).get("hits", [])
            results = []

            for hit in hits:
                document = hit.get("_source", {}).get("document", "")
                es_score = hit.get("_score", 0)

                # 对于SQL文档，需要解析JSON
                if collection_type == "sql" and document:
                    try:
                        doc_obj = json.loads(document)
                        doc_obj["es_score"] = es_score
                        results.append(doc_obj)
                    except json.JSONDecodeError:
                        results.append({"document": document, "es_score": es_score})
                else:
                    results.append(document)

            return results
        except Exception as e:
            logger.error(f"从ES搜索时出错: {str(e)}")
            return []

    def _fuse_results(self, dense_results: list, bm25_results: list, query: str = "") -> list:
        """使用上下文感知的融合方法融合向量搜索和BM25搜索结果"""
        if not bm25_results:
            return dense_results

        if not dense_results:
            return bm25_results

        method = self.fusion_method.lower()

        if method == "rrf":
            if not query:  # 没有查询上下文时使用基本RRF
                return RankFusion.reciprocal_rank_fusion(
                    [dense_results, bm25_results],
                    weights=[self.fusion_weights["vector"], self.fusion_weights["bm25"]]
                )
            else:  # 有查询上下文时使用上下文感知融合
                return RankFusion.contextual_fusion(
                    query=query,
                    dense_results=dense_results,
                    lexical_results=bm25_results,
                    metadata=None
                )
        elif method == "contextual":
            return RankFusion.contextual_fusion(
                query=query,
                dense_results=dense_results,
                lexical_results=bm25_results,
                metadata=None
            )
        elif method == "multi_stage":
            # 为简化实现，这里使用闭包封装搜索函数
            search_functions = {
                "vector": (lambda q: self._get_vector_results(q, collection_type="sql"), self.fusion_weights["vector"]),
                "bm25": (lambda q: self._search_es("sql", q, size=self.n_results_sql), self.fusion_weights["bm25"])
            }
            return RankFusion.multi_stage_fusion(query, search_functions)
        else:
            # 默认使用基础RRF
            return RankFusion.reciprocal_rank_fusion(
                [dense_results, bm25_results],
                weights=[self.fusion_weights["vector"], self.fusion_weights["bm25"]]
            )

    # 添加新的辅助方法来简化向量检索结果获取
    def _get_vector_results(self, query: str, collection_type: str) -> list:
        """获取向量检索结果的辅助方法"""
        embedding = self.generate_embedding(query)

        if collection_type == "sql":
            results = self._client.query_points(
                self.sql_collection_name,
                query=embedding,
                limit=self.n_results_sql,
                with_payload=True
            ).points

            return [dict(result.payload) for result in results]

        elif collection_type == "ddl":
            results = self._client.query_points(
                self.ddl_collection_name,
                query=embedding,
                limit=self.n_results_ddl,
                with_payload=True
            ).points

            return [result.payload.get("ddl", "") for result in results]

        elif collection_type == "documentation":
            results = self._client.query_points(
                self.documentation_collection_name,
                query=embedding,
                limit=self.n_results_documentation,
                with_payload=True
            ).points

            return [result.payload.get("documentation", "") for result in results]

        return []

    # 重写添加方法以同时添加到ES
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = super().add_question_sql(question, sql, **kwargs)

        # 同时索引到ES
        self._index_to_es("sql", question_sql_json, id)

        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = super().add_ddl(ddl, **kwargs)

        # 同时索引到ES
        self._index_to_es("ddl", ddl, id)

        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        添加文档到向量存储中，使用语义感知的分块策略

        Args:
            documentation: 要添加的文档内容
            **kwargs: 额外参数

        Returns:
            str: 文档ID列表的第一个ID
        """
        # 检查是否需要分块
        if len(documentation) > self.config.get("chunk_threshold", 500):
            # 创建分块器实例
            chunker = DocumentChunker(
                mechanical_terms_path=self.config.get("mechanical_terms_path", "/dictionary/MechanicalWords.txt"),
                max_chunk_size=self.config.get("max_chunk_size", 1000),
                overlap=self.config.get("chunk_overlap", 50)
            )

            # 分块
            chunks = chunker.chunk_document(documentation)

            # 对每个分块进行处理
            doc_ids = []
            for chunk in chunks:
                # 调用父类方法添加文档
                id = super().add_documentation(chunk, **kwargs)
                # 同时索引到ES
                self._index_to_es("documentation", chunk, id)
                doc_ids.append(id)

            # 返回第一个ID
            return doc_ids[0] if doc_ids else ""
        else:
            # 文档较小，不需要分块
            id = super().add_documentation(documentation, **kwargs)
            self._index_to_es("documentation", documentation, id)
            return id

    # 重写删除方法以同时从ES删除
    def remove_training_data(self, id: str, **kwargs) -> bool:
        result = super().remove_training_data(id, **kwargs)

        # 从ES中删除
        if result and self.es_client:
            collection_type = None
            if id.endswith("-sql"):
                collection_type = "sql"
            elif id.endswith("-ddl"):
                collection_type = "ddl"
            elif id.endswith("-doc"):
                collection_type = "documentation"

            if collection_type:
                index_name = self.es_indexes.get(collection_type)
                if index_name:
                    try:
                        self.es_client.delete(index=index_name, id=id)
                    except Exception as e:
                        logger.error(f"从ES删除文档时出错: {str(e)}")

        return result

    # 重写搜索方法以使用增强的融合算法
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of similar question-SQL pairs with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of question-SQL pairs sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "sql")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("sql", enhanced_question, size=self.n_results_sql)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"SQL retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No SQL results found for question: {question}")
                return []

            # 3. Apply reranking if enabled
            if use_rerank:
                # Combine candidates from both sources for reranking
                seen_ids = set()
                initial_candidates = []

                for result in dense_results + bm25_results:
                    # For SQL documents which are dictionaries with 'question' and 'sql'
                    if isinstance(result, dict):
                        result_id = json.dumps((result.get('question', ''), result.get('sql', '')), sort_keys=True)
                    else:
                        result_id = str(result)

                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        initial_candidates.append(result)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_sql_candidates(question, initial_candidates)

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=bm25_results,  # Keep original lexical results for diversity
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"Error in get_similar_question_sql: {str(e)}")
            logger.error(traceback.format_exc())
            # Return whatever results we have so far
            return dense_results or bm25_results or []

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of DDL statements with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of DDL statements sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # Add table name extraction for better DDL matching
            table_names = self._extract_table_names(question)
            if table_names:
                for name in table_names:
                    if name not in enhanced_question:
                        enhanced_question += f" {name}"

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "ddl")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("ddl", enhanced_question, size=self.n_results_ddl)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"DDL retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No DDL results found for question: {question}")
                return []

            # 3. Apply reranking if enabled
            if use_rerank:
                # Combine candidates from both sources for reranking
                seen_ddls = set()
                initial_candidates = []

                for ddl in dense_results + bm25_results:
                    ddl_str = str(ddl).strip()
                    if ddl_str and ddl_str not in seen_ddls:
                        seen_ddls.add(ddl_str)
                        initial_candidates.append(ddl_str)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_text_documents(question, initial_candidates, "ddl")

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=bm25_results,  # Keep original lexical results for diversity
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"Error in get_related_ddl: {str(e)}")
            logger.error(traceback.format_exc())
            # Return whatever results we have so far
            return dense_results or bm25_results or []

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        Enhanced retrieval of documentation with integrated reranking

        Args:
            question: The user's question
            **kwargs: Additional options including:
              - rerank: Whether to use reranking (default: True)
              - rerank_top_k: Number of candidates to rerank (default: 20)
              - fusion_method: Fusion method to use (default: self.fusion_method)

        Returns:
            List of documentation chunks sorted by relevance
        """
        try:
            # Extract options from kwargs with defaults
            use_rerank = kwargs.get('rerank', True)
            rerank_top_k = kwargs.get('rerank_top_k', 20)
            fusion_method = kwargs.get('fusion_method', self.fusion_method)

            # Start with enhanced question preprocessing
            enhanced_question = self.preprocess_field_names(question)

            # Extract business entities and domain terms to enhance docs retrieval
            domain_terms = self._extract_domain_terms(question)
            if domain_terms:
                for term in domain_terms:
                    if term not in enhanced_question:
                        enhanced_question += f" {term}"

            # 1. Vector search (using original question for semantic relevance)
            vector_start = time.time()
            dense_results = self._get_vector_results(question, "documentation")
            vector_time = time.time() - vector_start

            # 2. BM25 search (using enhanced question for better lexical matching)
            bm25_start = time.time()
            bm25_results = self._search_es("documentation", enhanced_question, size=self.n_results_documentation)
            bm25_time = time.time() - bm25_start

            # Log retrieval statistics
            logger.debug(f"Documentation retrieval - Vector: {len(dense_results)} results in {vector_time:.3f}s, "
                         f"BM25: {len(bm25_results)} results in {bm25_time:.3f}s")

            # Early return if no results from either method
            if not dense_results and not bm25_results:
                logger.warning(f"No documentation results found for question: {question}")
                return []

            # 3. Apply reranking if enabled
            if use_rerank:
                # Combine candidates from both sources for reranking
                seen_docs = set()
                initial_candidates = []

                for doc in dense_results + bm25_results:
                    doc_str = str(doc).strip()
                    if doc_str and doc_str not in seen_docs:
                        seen_docs.add(doc_str)
                        initial_candidates.append(doc_str)

                    if len(initial_candidates) >= rerank_top_k:
                        break

                # 4. Perform direct reranking
                reranked_results = self._rerank_text_documents(question, initial_candidates, "documentation")

                # 5. Apply final contextual fusion if needed
                if fusion_method == "contextual" and reranked_results:
                    final_results = RankFusion.contextual_fusion(
                        query=question,
                        dense_results=reranked_results,  # Use reranked results as dense results
                        lexical_results=bm25_results,  # Keep original lexical results for diversity
                        metadata={"vector_time": vector_time, "bm25_time": bm25_time}
                    )

                    if final_results:
                        return final_results

                # If reranking was successful, return the reranked results
                if reranked_results:
                    return reranked_results

            # 6. Fall back to regular fusion if reranking wasn't used or failed
            return self._fuse_results(dense_results, bm25_results, question)

        except Exception as e:
            logger.error(f"Error in get_related_documentation: {str(e)}")
            logger.error(traceback.format_exc())
            # Return whatever results we have so far
            return dense_results or bm25_results or []

    # 添加便利方法，预处理字段名以提高匹配质量
    def preprocess_field_names(self, question: str) -> str:
        """
        预处理问题中的字段名引用，以提高匹配质量
        """
        # 1. 检测可能的字段名（使用引号或驼峰/下划线分隔的词）
        potential_fields = re.findall(r'["\']([^"\']+)["\']|([a-z]+[A-Z][a-zA-Z]*)|([a-z_]+)', question)

        # 展平结果并过滤空项
        potential_fields = [field for group in potential_fields for field in group if field]

        # 2. 为每个潜在字段创建变体
        enhanced_question = question
        for field in potential_fields:
            # 创建常见的字段名变体
            variants = []

            # 驼峰转下划线
            if any(c.isupper() for c in field):
                snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in field]).lstrip('_')
                variants.append(snake_case)

            # 下划线转驼峰
            if '_' in field:
                camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(field.split('_')))
                variants.append(camel_case)

            # 构造增强后的问题
            for variant in variants:
                if variant not in question:
                    enhanced_question += f" {variant}"

        return enhanced_question

    def _patched_train(self, question: str = None, sql: str = None, ddl: str = None, documentation: str = None,
                       plan=None):
        """
        重写train方法，确保所有添加的训练数据都同步到ES
        """
        logger.info(f"使用增强版train方法，将同步数据到Elasticsearch")

        # 调用原始train方法获取结果
        result = self._original_train(question, sql, ddl, documentation, plan)

        # 确保数据已同步到ES
        if ddl:
            # ddl已经由add_ddl()方法同步，无需额外操作
            pass
        elif documentation:
            # documentation已经由add_documentation()方法同步，无需额外操作
            pass
        elif sql and question:
            # question和sql已经由add_question_sql()方法同步，无需额外操作
            pass
        elif plan:
            # 对于plan，需确保每项内容都同步到ES
            # 注意：这是防御性编程，通常plan.train()已经调用了各个add_*方法
            if self.es_client:
                try:
                    # 执行同步刷新，确保所有文档都已写入ES
                    for index_name in self.es_indexes.values():
                        self.es_client.indices.refresh(index=index_name)
                    logger.info("已刷新所有ES索引")
                except Exception as e:
                    logger.error(f"刷新ES索引时出错: {str(e)}")

        return result

    def get_sql_prompt(
            self,
            initial_prompt: str,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ):
        """增强版SQL提示词生成，提供更好的上下文和指导"""
        if initial_prompt is None:
            initial_prompt = f"你是一位精通{self.dialect}的专家。请基于提供的上下文为以下问题生成SQL查询。\n\n" + \
                             "请特别注意以下几点：\n" + \
                             "1. 适当处理NULL值\n" + \
                             "2. 当涉及多个表时使用正确的JOIN条件\n" + \
                             "3. 使用精确的列名和表引用\n" + \
                             "4. 考虑使用表别名提高可读性\n" + \
                             "5. 应用适当的排序、筛选和聚合函数\n" + \
                             "6. 正确处理字符串比较，考虑大小写敏感性\n" + \
                             "7. 日期时间处理需格式正确\n\n" + \
                             "你的回答应该仅基于给定的上下文，并遵循响应指南。"

        # 增强问题中的字段名称匹配，提高检索质量
        enhanced_question = self.preprocess_field_names(question)

        # 添加DDL和文档上下文
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        # 尝试从DDL提取表关系信息
        table_relationships = self._extract_table_relationships(ddl_list)
        if table_relationships:
            doc_list.append("===表关系信息\n" + table_relationships)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        # 添加更具体的响应指南
        initial_prompt += (
            "===响应指南\n"
            "1. 如果提供的上下文足够，请直接生成有效的SQL查询，无需解释。\n"
            "2. 如果上下文几乎足够但需要特定列中的具体值，请生成一个中间SQL查询来找出该列中的不同值。在查询前添加注释 intermediate_sql\n"
            "3. 如果上下文不足，请解释无法生成的原因。\n"
            "4. 请使用最相关的表。如果需要多个表，确保使用正确的JOIN条件。\n"
            "5. 如果问题之前已被问及并回答，请完全按照之前的方式重复答案。\n"
            f"6. 确保输出的SQL符合{self.dialect}规范，可执行且没有语法错误。\n"
            "7. 当结果需要排序时，使用适当的排序方法。\n"
            "8. 处理文本字段时，考虑大小写敏感性和可能的通配符。\n"
            "9. 处理日期时，确保正确的日期格式和比较方法。\n"
            "10. 对于聚合查询，确保GROUP BY子句包含所有非聚合列。\n"
        )

        message_log = [self.system_message(initial_prompt)]

        # 添加示例
        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        # 添加当前问题
        message_log.append(self.user_message(question))

        return message_log

    def _extract_table_relationships(self, ddl_list):
        """从DDL语句中提取表关系信息，兼容不同数据库"""
        if not ddl_list:
            return ""

        relationships = []

        # 更健壮的正则表达式，可以处理带有CONSTRAINT的定义
        constraint_pattern = r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+([^(]+)\s*\(([^)]+)\)'

        for ddl in ddl_list:
            fk_matches = re.findall(constraint_pattern, ddl, re.IGNORECASE)

            for match in fk_matches:
                fk_column = match[0].strip()
                ref_table = match[1].strip()
                ref_column = match[2].strip()

                # 清理表名中可能的引号和模式前缀
                ref_table = re.sub(r'["\'\`]', '', ref_table)  # 移除引号
                ref_table = ref_table.split('.')[-1]  # 提取模式限定符后的表名

                # 尝试获取当前表名
                table_match = re.search(r'CREATE TABLE\s+(?:["\'\`])?([^\s"\'`(]+)', ddl, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1).strip()
                    # 清理表名中可能的引号和模式前缀
                    table_name = re.sub(r'["\'\`]', '', table_name)
                    table_name = table_name.split('.')[-1]

                    relationships.append(f"表 {table_name} 的列 {fk_column} 引用 表 {ref_table} 的列 {ref_column}")

        return "\n".join(relationships)

    def _extract_table_names(self, question: str) -> List[str]:
        """
        Extract potential table names from a question

        Args:
            question: The user's question

        Returns:
            List of potential table names
        """
        table_names = []

        # Pattern for common table name references
        patterns = [
            r'(?:table|tables|from|join)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "from users" or "join orders"
            r'(?:in|on|for)\s+the\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?\s+table',  # "in the users table"
            r'(?:data|records|rows|information)\s+(?:from|in|of)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "data from users"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            table_names.extend([match.lower() for match in matches if match])

        # Include common pluralization/singularization
        expanded_names = set(table_names)
        for name in table_names:
            # Add singular version if plural
            if name.endswith('s'):
                expanded_names.add(name[:-1])
            # Add plural version if singular
            else:
                expanded_names.add(name + 's')

        return list(expanded_names)

    def _extract_domain_terms(self, question: str) -> List[str]:
        """
        Extract business domain terms from a question

        Args:
            question: The user's question

        Returns:
            List of business domain terms
        """
        domain_terms = []

        # Common business entities and their synonyms
        business_entities = {
            "订单": ["order", "purchase", "transaction"],
            "客户": ["customer", "client", "user", "buyer"],
            "产品": ["product", "item", "goods", "merchandise"],
            "支付": ["payment", "transaction", "money", "fund"],
            "价格": ["price", "cost", "amount", "value"],
            "状态": ["status", "state", "condition", "phase"],
            "时间": ["time", "date", "period", "duration"],
            "地址": ["address", "location", "place", "destination"],
            "类型": ["type", "category", "class", "kind"],
            "数量": ["quantity", "amount", "number", "count"]
        }

        # Check for presence of business entities
        for entity, synonyms in business_entities.items():
            if entity in question:
                domain_terms.extend(synonyms)

        # Extract potential technical terms
        tech_patterns = [
            r'(?:table|column|field)\s+[\'"`]?([a-zA-Z0-9_]+)[\'"`]?',  # "field 'user_id'"
            r'(?:aggregat\w+|calculat\w+|comput\w+)\s+(?:the\s+)?([a-zA-Z0-9_]+)',  # "calculate the total_price"
            r'(?:sort|order|group)\s+by\s+([a-zA-Z0-9_]+)',  # "group by region"
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            domain_terms.extend([match.lower() for match in matches if match])

        return list(set(domain_terms))

    def _rerank_sql_candidates(self, query: str, candidates: List[Any], timeout: int = 10) -> List[Any]:
        """
        Rerank SQL candidates using the reranking service

        Args:
            query: The user's question
            candidates: List of candidate SQL items (dicts with 'question' and 'sql')
            timeout: Timeout for the reranking request in seconds

        Returns:
            Reranked list of candidates or empty list if reranking failed
        """
        if not candidates:
            return []

        # Check if reranking is available
        rerank_url = self.config.get("rerank_url", "http://localhost:8091")

        try:
            # Prepare documents for reranking
            documents = []
            original_items = []

            for candidate in candidates:
                if isinstance(candidate, dict):
                    # Format SQL examples for reranking
                    doc_content = f"Question: {candidate.get('question', '')}\nSQL: {candidate.get('sql', '')}"
                    documents.append({"content": doc_content})
                    original_items.append(candidate)
                else:
                    # Handle non-dict candidates (shouldn't happen for SQL but added for safety)
                    documents.append({"content": str(candidate)})
                    original_items.append(candidate)

            # Call reranking service
            response = requests.post(
                rerank_url,
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": len(documents)  # Rerank all candidates
                },
                timeout=timeout
            )

            if response.status_code == 200:
                # Process reranking results
                reranked_data = response.json()
                reranked_results = []

                # Reconstruct original items in new order with scores
                for item in reranked_data.get("results", []):
                    idx = item.get("index")
                    if 0 <= idx < len(original_items):
                        # If the original item is a dict, add the reranking score
                        if isinstance(original_items[idx], dict):
                            item_copy = original_items[idx].copy()
                            item_copy["rerank_score"] = item.get("score", 0)
                            reranked_results.append(item_copy)
                        else:
                            reranked_results.append(original_items[idx])

                logger.debug(f"Reranked {len(reranked_results)} SQL candidates")
                return reranked_results
            else:
                logger.warning(f"Reranking service returned error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error during SQL reranking: {str(e)}")
            return []

    def _rerank_text_documents(self, query: str, documents: List[str], doc_type: str = "documentation",
                               timeout: int = 10) -> List[str]:
        """
        Rerank text documents using the reranking service

        Args:
            query: The user's question
            documents: List of document strings to rerank
            doc_type: Type of documents ("ddl" or "documentation")
            timeout: Timeout for the reranking request in seconds

        Returns:
            Reranked list of documents or empty list if reranking failed
        """
        if not documents:
            return []

        # Check if reranking is available
        rerank_url = self.config.get("rerank_url", "http://localhost:8091")

        try:
            # Prepare request payload
            # For DDL documents, we might want to extract table names for the reranking query
            enhanced_query = query
            if doc_type == "ddl":
                table_names = self._extract_table_names(query)
                if table_names:
                    enhanced_query = f"{query} table:" + " table:".join(table_names)

            # Call reranking service
            response = requests.post(
                rerank_url,
                json={
                    "query": enhanced_query,
                    "documents": [{"content": doc} for doc in documents],
                    "top_k": len(documents)  # Rerank all documents
                },
                timeout=timeout
            )

            if response.status_code == 200:
                # Process reranking results
                reranked_data = response.json()
                reranked_results = []

                # Reconstruct original documents in new order
                for item in reranked_data.get("results", []):
                    idx = item.get("index")
                    if 0 <= idx < len(documents):
                        reranked_results.append(documents[idx])

                logger.debug(f"Reranked {len(reranked_results)} {doc_type} documents")
                return reranked_results
            else:
                logger.warning(f"Reranking service returned error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error during {doc_type} reranking: {str(e)}")
            return []


from vanna.flask import VannaFlaskApp


# 扩展VannaFlaskApp类来添加自定义API端点
class EnhancedVannaFlaskApp(VannaFlaskApp):
    def __init__(self, vn, *args, **kwargs):
        super().__init__(vn, *args, **kwargs)

        # 存储检索服务实例
        self.retrieval_service = retrieval_service

        # Initialize task manager
        self.task_manager = AsyncTaskManager(max_workers=5)

        # 添加自定义API端点
        self.add_custom_endpoints()
        # 添加数据管理API端点
        self._add_data_management_endpoints()

    def add_custom_endpoints(self):
        # 添加检索上下文API
        @self.flask_app.route("/api/v0/get_retrieval_context", methods=["GET"])
        @self.requires_auth
        def get_retrieval_context(user: any):
            try:
                # 获取请求参数
                question = request.args.get("question")
                max_results = int(request.args.get("max_results", "10"))
                include_prompt = request.args.get("include_prompt", "true").lower() == "true"
                enhance_query = request.args.get("enhance_query", "true").lower() == "true"
                use_rerank = request.args.get("use_rerank", "true").lower() == "true"

                if not question:
                    return jsonify({"type": "error", "error": "No question provided"}), 400

                # 使用检索服务执行检索
                options = {
                    "max_results": max_results,
                    "enhance_query": enhance_query,
                    "use_rerank": use_rerank
                }

                retrieval_results = self.retrieval_service.retrieve(question, options)

                # 生成提示词（如需要）
                prompt = None
                if include_prompt:
                    prompt_start = time.time()
                    prompt = self.vn.get_sql_prompt(
                        initial_prompt=None,
                        question=question,
                        question_sql_list=retrieval_results["question_sql_list"],
                        ddl_list=retrieval_results["ddl_list"],
                        doc_list=retrieval_results["doc_list"]
                    )
                    retrieval_results["stats"]["prompt_generation_time"] = round(time.time() - prompt_start, 3)

                # 构建响应
                response = {
                    "type": "retrieval_context",
                    "question": question,
                    "context": {
                        "question_sql_list": utils.serialize_data(retrieval_results["question_sql_list"]),
                        "ddl_list": retrieval_results["ddl_list"],
                        "doc_list": retrieval_results["doc_list"],
                        "table_relationships": retrieval_results["table_relationships"]
                    },
                    "dialect": getattr(self.vn, "dialect", "MySQL"),
                    "retrieval_stats": retrieval_results["stats"],
                    "total_time": retrieval_results["total_time"]
                }

                if include_prompt:
                    response["prompt"] = prompt

                return jsonify(response)

            except Exception as e:
                logger.error(f"检索API错误: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "type": "error",
                    "error": str(e),
                    "details": traceback.format_exc() if hasattr(self.vn, "debug") and self.vn.debug else None
                }), 500

        # 添加执行外部SQL的API
        @self.flask_app.route("/api/v0/execute_external_sql", methods=["POST"])
        @self.requires_auth
        def execute_external_sql(user: any):
            try:
                data = request.json
                question = data.get("question")
                sql = data.get("sql")

                if not sql:
                    return jsonify({"type": "error", "error": "No SQL provided"}), 400

                # 生成ID并存储问题和SQL
                id = self.cache.generate_id(question=question)
                self.cache.set(id=id, field="question", value=question)
                self.cache.set(id=id, field="sql", value=sql)

                # 执行SQL
                try:
                    df = self.vn.run_sql(sql=sql)
                    self.cache.set(id=id, field="df", value=df)

                    # 返回结果
                    return jsonify({
                        "type": "df",
                        "id": id,
                        "df": df.head(10).to_json(orient='records', date_format='iso'),
                        "should_generate_chart": self.vn.should_generate_chart(df),
                    })
                except Exception as e:
                    return jsonify({"type": "sql_error", "error": str(e)})

            except Exception as e:
                logger.error(f"执行SQL API错误: {str(e)}")
                return jsonify({"type": "error", "error": str(e)}), 500

    def _add_data_management_endpoints(self):
            """添加数据管理相关的API端点"""

            # 查看提交任务状态
            @self.flask_app.route("/api/v0/task_status", methods=["GET"])
            @self.requires_auth
            def get_task_status(user: any):
                try:
                    task_id = request.args.get("task_id")
                    if not task_id:
                        return jsonify({"type": "error", "error": "No task_id provided"}), 400

                    status = self.task_manager.get_task_status(task_id)
                    return jsonify({
                        "type": "task_status",
                        "task_id": task_id,
                        "status": status
                    })
                except Exception as e:
                    logger.error(f"Task status error: {str(e)}")
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传MD文档
            @self.flask_app.route("/api/v0/upload_documentation", methods=["POST"])
            @self.requires_auth
            def upload_documentation(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.md')):
                        return jsonify({"type": "error", "error": "文件类型必须是Markdown"}), 400

                    # Read file content
                    file_content = file.read().decode('utf-8')

                    # Submit document processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_document,
                        callback=lambda result: logger.info(
                            f"Document training completed: {len(result) if isinstance(result, list) else 'single'} chunks processed"),
                        file_content=file_content,
                        file_type=file.filename.split('.')[-1]
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "文档上传已开始。系统将在后台处理文档分块和训练。"
                    })
                except Exception as e:
                    logger.error(f"上传文档错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传DDL文件
            @self.flask_app.route("/api/v0/upload_ddl", methods=["POST"])
            @self.requires_auth
            def upload_ddl(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.sql', '.ddl', '.txt')):
                        return jsonify({"type": "error", "error": "文件类型必须是SQL、DDL或TXT"}), 400

                    # Read file content
                    file_content = file.read().decode('utf-8')

                    # Submit DDL processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_ddl,
                        callback=lambda result: logger.info(
                            f"DDL training completed: {result['success']} of {result['total']} processed successfully"),
                        file_content=file_content
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "DDL上传已开始。系统将在后台提取和训练表定义。"
                    })
                except Exception as e:
                    logger.error(f"上传DDL错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

            # 上传SQL示例
            @self.flask_app.route("/api/v0/upload_sql_examples", methods=["POST"])
            @self.requires_auth
            def upload_sql_examples(user: any):
                try:
                    # Check if file exists
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # Check file type
                    if not file.filename.endswith(('.xlsx', '.csv')):
                        return jsonify({"type": "error", "error": "文件类型必须是Excel或CSV"}), 400

                    # Save file temporarily
                    temp_file = os.path.join('/tmp', file.filename)
                    file.save(temp_file)

                    # Submit SQL examples processing as an async task
                    task_id = self.task_manager.submit_task(
                        self._process_sql_examples,
                        callback=lambda result: logger.info(
                            f"SQL examples training completed: {result['success']} of {result['total']} processed successfully"),
                        file_path=temp_file,
                        file_type=file.filename.split('.')[-1]
                    )

                    return jsonify({
                        "type": "task_submitted",
                        "task_id": task_id,
                        "message": "SQL示例上传已开始。系统将在后台处理和训练示例。"
                    })
                except Exception as e:
                    logger.error(f"上传SQL示例错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500
            # 上传术语词典
            @self.flask_app.route("/api/v0/upload_terminology", methods=["POST"])
            @self.requires_auth
            def upload_terminology(user: any):
                try:
                    # 检查是否有文件
                    if 'file' not in request.files:
                        return jsonify({"type": "error", "error": "未找到文件"}), 400

                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"type": "error", "error": "未选择文件"}), 400

                    # 检查文件类型
                    if not file.filename.endswith(('.txt', '.xlsx', '.csv')):
                        return jsonify({"type": "error", "error": "文件类型必须是TXT、Excel或CSV"}), 400

                    # 读取并保存文件
                    mechanical_terms_path = os.path.join('/dictionary', 'Mechanical_words.txt')
                    os.makedirs(os.path.dirname(mechanical_terms_path), exist_ok=True)

                    # 处理不同格式
                    if file.filename.endswith('.txt'):
                        file.save(mechanical_terms_path)
                        term_count = sum(1 for _ in open(mechanical_terms_path, 'r', encoding='utf-8').readlines())
                        term_count = term_count // 2  # 一行英文一行中文，所以除以2
                    else:
                        # 对于Excel或CSV，需要转换格式
                        temp_file = os.path.join('/tmp', file.filename)
                        file.save(temp_file)

                        if file.filename.endswith('.xlsx'):
                            df = pd.read_excel(temp_file)
                        else:
                            df = pd.read_csv(temp_file)

                        # 检查必要的列
                        required_columns = ['english', 'chinese']
                        if not all(col in df.columns for col in required_columns):
                            return jsonify({"type": "error", "error": "文件必须包含'english'和'chinese'列"}), 400

                        # 写入到txt文件，一行英文一行中文
                        with open(mechanical_terms_path, 'w', encoding='utf-8') as f:
                            for _, row in df.iterrows():
                                if pd.notna(row['english']) and pd.notna(row['chinese']):
                                    f.write(f"{row['english'].strip()}\n{row['chinese'].strip()}\n")

                        term_count = len(df)
                        os.remove(temp_file)

                    return jsonify({
                        "type": "terminology_upload_result",
                        "path": mechanical_terms_path,
                        "terms": term_count,
                        "status": "success"
                    })
                except Exception as e:
                    logger.error(f"上传术语词典错误: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"type": "error", "error": str(e)}), 500

                # 辅助方法: 按Markdown标题分块

    # 异步处理DDL文档
    def _process_ddl(self, file_content):
        """
        Process DDL content by extracting CREATE TABLE statements
        and training them one by one

        Args:
            file_content: The content of the DDL file

        Returns:
            Dict with processing results
        """
        try:
            # Split into DDL statements
            ddl_statements = re.findall(r'CREATE\s+TABLE\s+.*?;', file_content, re.DOTALL | re.IGNORECASE)

            if not ddl_statements:
                raise ValueError("未找到有效的CREATE TABLE语句")

            # Process each DDL statement
            results = []
            for ddl in ddl_statements:
                try:
                    # Extract table name for logging
                    table_name_match = re.search(r'CREATE\s+TABLE\s+(?:`|")?([^`"\s(]+)', ddl, re.IGNORECASE)
                    table_name = table_name_match.group(1) if table_name_match else "未知表"

                    # Train DDL
                    id = self.vn.train(ddl=ddl)
                    results.append({"table": table_name, "id": id, "status": "success"})
                except Exception as e:
                    results.append({"table": table_name, "error": str(e), "status": "error"})

            return {
                "total": len(ddl_statements),
                "success": sum(1 for r in results if r["status"] == "success"),
                "results": results
            }
        except Exception as e:
            logger.error(f"DDL处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

    # 异步处理SQL问题对文档
    def _process_sql_examples(self, file_path, file_type):
        """
        Process SQL examples from Excel or CSV file

        Args:
            file_path: Path to the temporary file
            file_type: File extension (xlsx or csv)

        Returns:
            Dict with processing results
        """
        try:
            # Read file based on type
            if file_type == 'xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Clean up temp file
            try:
                os.remove(file_path)
            except:
                pass

            # Check required columns
            required_columns = ['question', 'sql']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("文件必须包含'question'和'sql'列")

            # Process each row
            results = []
            for i, row in df.iterrows():
                try:
                    if pd.isna(row['question']) or pd.isna(row['sql']):
                        continue

                    question = str(row['question']).strip()
                    sql = str(row['sql']).strip()

                    if not question or not sql:
                        continue

                    # Train question/SQL pair
                    id = self.vn.train(question=question, sql=sql)
                    results.append({
                        "row": i + 1,
                        "question": question[:50] + "..." if len(question) > 50 else question,
                        "id": id,
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "row": i + 1,
                        "question": question[:50] + "..." if len(question) > 50 else question,
                        "error": str(e),
                        "status": "error"
                    })

            return {
                "total": len(df),
                "success": sum(1 for r in results if r["status"] == "success"),
                "results": results
            }
        except Exception as e:
            logger.error(f"SQL示例处理错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

    # 异步处理MD文档
    def _process_document(self, file_content, file_type):
            """
            处理文档内容

            Args:
                file_content: 文件内容
                file_type: 文件类型(md或txt)

            Returns:
                文档ID列表
            """
            try:
                # 直接将整个文档传递给train方法，让它内部处理分块
                doc_id = self.vn.train(documentation=file_content)
                return [doc_id]  # 返回ID列表，方便以后扩展
            except Exception as e:
                logger.error(f"文档处理错误: {str(e)}")
                logger.error(traceback.format_exc())
                raise e
# 主程序
if __name__ == "__main__":
    # 初始化增强版Vanna - 连接到远程ChromaDB和ES
    vn = EnhancedVanna(client=client, config={
        'model': 'qwen-plus',
        'qdrant_url': 'http://124.71.225.73:6333',  # 修改为您的Qdrant服务地址
        'elasticsearch': {
            'hosts': ['http://192.168.66.100:9200']
        },
        'fusion_method': 'contextual',  # 使用上下文感知融合，可选值: rrf, borda, contextual, multi_stage
        'vector_weight': 0.6,  # 向量检索结果权重
        'bm25_weight': 0.4  # BM25检索结果权重
    })

    # 连接到MySQL
    vn.connect_to_mysql(
        host=os.getenv("MYSQL_HOST", "124.71.225.73"),
        dbname=os.getenv("MYSQL_DATABASE", "testdb"),
        user=os.getenv("MYSQL_USER", "testuser"),
        password=os.getenv("MYSQL_PASSWORD", "testpassword"),
        port=int(os.getenv("MYSQL_PORT", "3306"))
    )

    # 初始化检索服务（只创建一次）
    retrieval_service = RetrievalService(vn, {
        "rerank_url": "http://localhost:8091",
        "rerank_enabled": True,
        "max_results": 10
    })

    app = EnhancedVannaFlaskApp(vn)
    app.run(host="0.0.0.0", port=8084)