# 使用官方 Python 镜像作为基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/models \
    TZ=Asia/Shanghai

# 设置 pip 使用阿里云的镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装系统依赖（对elasticsearch和numpy可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install \
    vanna[chromadb,postgres,ollama,mysql,qdrant,openai] \
    elasticsearch \
    sentence-transformers==3.4.1 \
    transformers==4.49.0 \
    accelerate<=0.23.0 \
    qdrant-client \
    openpyxl \
    xlrd \
    pandas \
    requests \
    Flask

# 创建必要的目录
RUN mkdir -p ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_m3e-base \
    && mkdir -p ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_bge-m3 \
    && mkdir -p /dictionary

# 创建词典目录
RUN touch /dictionary/MechanicalWords.txt

COPY ./MechanicalWords.txt /dictionary/MechanicalWords.txt

# 复制模型到指定路径
COPY ./m3e-base  ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_m3e-base
COPY ./bge-m3  ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_bge-m3

# 暴露端口
EXPOSE 8084


# 执行容器时默认运行的命令
CMD ["python", "app.py"]
