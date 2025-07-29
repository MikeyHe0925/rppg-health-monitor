# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码和模型
COPY app /app/app
COPY models /app/models

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "app.app:app"]