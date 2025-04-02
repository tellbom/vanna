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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVanna")

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="sk-9f8124e18aa242af830c8a502c015c40",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


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
        self.rerank_url = self.config.get("rerank_url", "http://192.168.66.100:8091/rerank")
        self.rerank_enabled = self.config.get("rerank_enabled", True)
        self.rerank_timeout = self.config.get("rerank_timeout", 10)

        # 检索配置
        self.max_results = self.config.get("max_results", 10)

        logger.info(f"检索服务初始化完成, 重排序{'启用' if self.rerank_enabled else '禁用'}")

    def retrieve(self, question: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行全面检索流程

        Args:
            question: 用户问题
            options: 检索选项，可包含:
                - max_results: 最大结果数量
                - enhance_query: 是否增强查询
                - use_rerank: 是否使用重排序

        Returns:
            Dict包含检索结果和统计信息
        """
        start_time = time.time()
        options = options or {}
        stats = {}

        # 提取选项
        max_results = options.get("max_results", self.max_results)
        enhance_query = options.get("enhance_query", True)
        use_rerank = options.get("use_rerank", self.rerank_enabled)

        # 步骤1: 增强查询 - 利用EnhancedVanna中的方法
        if enhance_query and hasattr(self.vn, "preprocess_field_names"):
            enhanced_question = self.vn.preprocess_field_names(question)
            stats["enhanced_question"] = enhanced_question
        else:
            enhanced_question = question

        # 步骤2: 检索SQL示例 - 利用EnhancedVanna的混合检索
        sql_start = time.time()
        question_sql_list = self.vn.get_similar_question_sql(question)
        stats["sql_retrieval_time"] = round(time.time() - sql_start, 3)
        stats["sql_results_count"] = len(question_sql_list)

        # 步骤3: 检索DDL - 利用EnhancedVanna的混合检索
        ddl_start = time.time()
        ddl_list = self.vn.get_related_ddl(question)
        stats["ddl_retrieval_time"] = round(time.time() - ddl_start, 3)
        stats["ddl_results_count"] = len(ddl_list)

        # 步骤4: 检索文档 - 利用EnhancedVanna的混合检索
        doc_start = time.time()
        doc_list = self.vn.get_related_documentation(question)
        stats["doc_retrieval_time"] = round(time.time() - doc_start, 3)
        stats["doc_results_count"] = len(doc_list)

        # 步骤5: 使用重排序服务优化结果 (只对DDL和文档进行重排序)
        if use_rerank and self.rerank_enabled:
            reranked_results = self._rerank_documents(question, ddl_list, doc_list, max_results)
            if reranked_results:
                ddl_list = reranked_results.get("ddl_list", ddl_list)
                doc_list = reranked_results.get("doc_list", doc_list)
                stats.update(reranked_results.get("stats", {}))

        # 步骤6: 提取表关系
        table_relationships = ""
        if hasattr(self.vn, "_extract_table_relationships"):
            table_relationships = self.vn._extract_table_relationships(ddl_list)

        # 步骤7: 限制结果数量
        if max_results > 0:
            if len(question_sql_list) > max_results:
                question_sql_list = question_sql_list[:max_results]
            if len(ddl_list) > max_results:
                ddl_list = ddl_list[:max_results]
            if len(doc_list) > max_results:
                doc_list = doc_list[:max_results]

        # 返回检索结果
        results = {
            "question_sql_list": question_sql_list,
            "ddl_list": ddl_list,
            "doc_list": doc_list,
            "table_relationships": table_relationships,
            "stats": stats,
            "total_time": round(time.time() - start_time, 3)
        }

        return results

    def _rerank_documents(self, question: str, ddl_list: List[str], doc_list: List[str],
                          top_k: int = None) -> Optional[Dict]:
        """使用重排序服务优化文档排序"""
        if not (ddl_list or doc_list):
            return None

        rerank_start = time.time()
        stats = {}

        try:
            # 准备文档
            documents = []

            # 添加DDL文档
            for ddl in ddl_list:
                documents.append({"content": ddl, "type": "ddl"})

            # 添加文档
            for doc in doc_list:
                documents.append({"content": doc, "type": "doc"})

            # 调用重排序服务
            response = requests.post(
                self.rerank_url,
                json={
                    "query": question,
                    "documents": documents,
                    "top_k": top_k or len(documents)
                },
                timeout=self.rerank_timeout
            )

            if response.status_code == 200:
                # 处理重排序结果
                reranked_results = response.json().get("results", [])

                # 分离不同类型的文档
                reranked_ddl = []
                reranked_docs = []

                for item in reranked_results:
                    if item.get("type") == "ddl":
                        reranked_ddl.append(item.get("content"))
                    elif item.get("type") == "doc":
                        reranked_docs.append(item.get("content"))

                stats["rerank_time"] = round(time.time() - rerank_start, 3)
                stats["reranked"] = True

                return {
                    "ddl_list": reranked_ddl if reranked_ddl else ddl_list,
                    "doc_list": reranked_docs if reranked_docs else doc_list,
                    "stats": stats
                }
            else:
                logger.error(f"重排序服务返回错误: {response.status_code}, {response.text}")
                stats["rerank_error"] = f"服务返回状态码: {response.status_code}"
                return None
        except Exception as e:
            logger.error(f"调用重排序服务出错: {str(e)}")
            stats["rerank_error"] = str(e)
            return None


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
    def contextual_fusion(query: str, dense_results: list, lexical_results: list, metadata=None, k: int = 60) -> list:
        """
        上下文感知的融合算法，针对相近语义字段优化

        Args:
            query: 用户查询
            dense_results: 向量检索结果
            lexical_results: 文本检索结果
            metadata: 元数据信息
            k: RRF常数

        Returns:
            融合后的结果列表
        """
        # 提取查询特征
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        is_status_query = any(term in query_terms for term in ['状态', '取消', '完成', '支付'])
        is_time_query = any(term in query_terms for term in ['时间', '日期', '年', '月', '日'])
        is_type_query = any(term in query_terms for term in ['类型', '种类', '分类'])

        # 动态调整权重
        if is_status_query or is_type_query:
            # 状态和类型查询，BM25可能更准确
            vector_weight = 0.4
            lexical_weight = 0.6
        elif is_time_query:
            # 时间查询，两者都重要
            vector_weight = 0.5
            lexical_weight = 0.5
        else:
            # 默认权重
            vector_weight = 0.7
            lexical_weight = 0.3

        # 计算融合分数
        scores = {}

        # 处理向量结果
        for rank, item in enumerate(dense_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            scores[item_key]["score"] += vector_weight * (1.0 / (k + rank))
            scores[item_key]["matches"].add("vector")

        # 处理文本结果
        for rank, item in enumerate(lexical_results, start=1):
            item_key = RankFusion._get_item_key(item)
            if item_key not in scores:
                scores[item_key] = {"item": item, "score": 0, "matches": set()}

            scores[item_key]["score"] += lexical_weight * (1.0 / (k + rank))
            scores[item_key]["matches"].add("lexical")

            # 额外的上下文奖励
            item_str = str(item)

            # 检查结果是否包含查询词
            term_matches = sum(1 for term in query_terms if term in item_str.lower())
            term_match_ratio = term_matches / len(query_terms) if query_terms else 0

            # 词匹配奖励
            scores[item_key]["score"] *= (1 + 0.2 * term_match_ratio)

        # 多检索源奖励
        for item_key, data in scores.items():
            if len(data["matches"]) > 1:
                data["score"] *= 1.25  # 同时出现在两种检索中的项获得25%的额外分数

        # 按分数排序
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


# 扩展的Vanna类 - 集成ES和向量数据库
class EnhancedVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        self.config = config or {}
        model_name = self.config.get("embedding_model", "bge-m3")

        # 设置嵌入函数
        class CompatibleEmbeddingFunction:
            def __init__(self, model_name: str):
                self.model = SentenceTransformer(
                    model_name_or_path=f"/models/sentence-transformers_{model_name}",
                    local_files_only=True
                )

            def __call__(self, input: List[str]) -> np.ndarray:
                # 直接返回 NumPy 数组
                return self.model.encode(input, convert_to_numpy=True)

        self.config["embedding_function"] = CompatibleEmbeddingFunction(model_name)

        # 设置 ChromaDB 远程连接
        if "chroma_host" in self.config and "chroma_port" in self.config:
            # 创建远程 ChromaDB 客户端
            chroma_client = chromadb.HttpClient(
                host=self.config.get("chroma_host"),
                port=self.config.get("chroma_port"),
                settings=Settings(anonymized_telemetry=False)
            )
            self.config["client"] = chroma_client

        # 初始化父类
        ChromaDB_VectorStore.__init__(self, config=self.config)
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
        elif method == "borda":
            return RankFusion.borda_count(
                [dense_results, bm25_results],
                weights=[self.fusion_weights["vector"], self.fusion_weights["bm25"]]
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
        if collection_type == "sql":
            return ChromaDB_VectorStore._extract_documents(
                self.sql_collection.query(
                    query_texts=[query],
                    n_results=self.n_results_sql,
                )
            )
        elif collection_type == "ddl":
            return ChromaDB_VectorStore._extract_documents(
                self.ddl_collection.query(
                    query_texts=[query],
                    n_results=self.n_results_ddl,
                )
            )
        elif collection_type == "documentation":
            return ChromaDB_VectorStore._extract_documents(
                self.documentation_collection.query(
                    query_texts=[query],
                    n_results=self.n_results_documentation,
                )
            )
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
        id = super().add_documentation(documentation, **kwargs)

        # 同时索引到ES
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
        """使用上下文感知融合的相似问题检索"""
        # 增强问题中的字段名称匹配
        enhanced_question = self.preprocess_field_names(question)

        # 向量搜索
        dense_results = self._get_vector_results(question, "sql")  # 使用原始问题进行向量检索

        # BM25搜索 - 使用增强后的问题提高匹配率
        bm25_results = self._search_es("sql", enhanced_question, size=self.n_results_sql)

        logger.debug(f"SQL检索 - 向量结果数: {len(dense_results)}, BM25结果数: {len(bm25_results)}")

        # 融合结果 - 传入原始问题以提供上下文
        return self._fuse_results(dense_results, bm25_results, question)

    def get_related_ddl(self, question: str, **kwargs) -> list:
        """使用上下文感知融合的DDL检索"""
        # 增强问题中的字段名称匹配
        enhanced_question = self.preprocess_field_names(question)

        # 向量搜索
        dense_results = self._get_vector_results(question, "ddl")

        # BM25搜索 - 使用增强后的问题提高匹配率
        bm25_results = self._search_es("ddl", enhanced_question, size=self.n_results_ddl)

        logger.debug(f"DDL检索 - 向量结果数: {len(dense_results)}, BM25结果数: {len(bm25_results)}")

        # 融合结果 - 传入原始问题以提供上下文
        return self._fuse_results(dense_results, bm25_results, question)

    def get_related_documentation(self, question: str, **kwargs) -> list:
        """使用上下文感知融合的文档检索"""
        # 增强问题中的字段名称匹配
        enhanced_question = self.preprocess_field_names(question)

        # 向量搜索
        dense_results = self._get_vector_results(question, "documentation")

        # BM25搜索 - 使用增强后的问题提高匹配率
        bm25_results = self._search_es("documentation", enhanced_question, size=self.n_results_documentation)

        logger.debug(f"文档检索 - 向量结果数: {len(dense_results)}, BM25结果数: {len(bm25_results)}")

        # 融合结果 - 传入原始问题以提供上下文
        return self._fuse_results(dense_results, bm25_results, question)

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

    # 启动Flask应用


from vanna.flask import VannaFlaskApp


# 扩展VannaFlaskApp类来添加自定义API端点
class EnhancedVannaFlaskApp(VannaFlaskApp):
    def __init__(self, vn, *args, **kwargs):
        super().__init__(vn, *args, **kwargs)

        # 存储检索服务实例
        self.retrieval_service = retrieval_service

        # 添加自定义API端点
        self.add_custom_endpoints()

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
                    logger.info(f"提示词信息: {prompt}")
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


# 主程序
if __name__ == "__main__":
    # 初始化增强版Vanna - 连接到远程ChromaDB和ES
    vn = EnhancedVanna(client=client, config={
        'model': 'qwen-plus',
        'chroma_host': '124.71.225.73',
        'chroma_port': 8000,
        'elasticsearch': {
            'hosts': ['http://192.168.66.100:9200']
        },
        'fusion_method': 'contextual',  # 使用上下文感知融合，可选值: rrf, borda, contextual, multi_stage
        'vector_weight': 0.6,  # 向量检索结果权重
        'bm25_weight': 0.4  # BM25检索结果权重
    })

    vn.train(ddl="""
        CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- 用户ID，自增长
    username VARCHAR(100) NOT NULL,  -- 用户名，最大长度100
    email VARCHAR(100),  -- 邮箱，最大长度100
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 创建时间，默认当前时间
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表，用于存储用户信息';

    """)

    vn.train(ddl="""
          CREATE TABLE IF NOT EXISTS orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,  -- 订单ID，自增长
    user_id INT NOT NULL,  -- 用户ID，关联用户表
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 订单日期，默认当前时间
    amount DECIMAL(10, 2) NOT NULL,  -- 订单金额，保留两位小数
    FOREIGN KEY (user_id) REFERENCES users(id)  -- 外键约束，关联用户表的id
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单表，用于存储订单信息';


        """)

    # 连接到MySQL
    vn.connect_to_mysql(
        host=os.getenv("MYSQL_HOST", "192.168.66.100"),
        dbname=os.getenv("MYSQL_DATABASE", "testdb"),
        user=os.getenv("MYSQL_USER", "testuser"),
        password=os.getenv("MYSQL_PASSWORD", "testpassword"),
        port=int(os.getenv("MYSQL_PORT", "3306"))
    )

    # 初始化检索服务（只创建一次）
    retrieval_service = RetrievalService(vn, {
        "rerank_url": "http://192.168.66.100:8091/rerank",
        "rerank_enabled": True,
        "max_results": 10
    })

    app = EnhancedVannaFlaskApp(vn)
    app.run(host="0.0.0.0", port=8084)