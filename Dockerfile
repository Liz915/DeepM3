# 使用轻量级 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 先拷贝依赖文件 (利用 Docker 缓存机制加速构建)
COPY requirements.txt .

# 2. 安装依赖
# --no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 拷贝项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
ENV PYTHONPATH=/app
CMD ["python", "src/api.py"]