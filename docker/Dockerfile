# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/pytorch:24.05-py3

# 這確保在運行 apt 命令時不會有任何用戶交互。
# 這是為了確保 Docker 映像建構過程自動進行，無需人工介入。
ENV DEBIAN_FRONTEND=noninteractive

# 防止 Python 創建 .pyc 字節碼文件。
ENV PYTHONDONTWRITEBYTECODE=1

# 為Matplotlib和Transformers建立配置和緩存目錄
RUN mkdir -p /app/matplotlib_config /app/transformers_cache

# 設置環境變量
ENV MPLCONFIGDIR /app/matplotlib_config
ENV TRANSFORMERS_CACHE /app/transformers_cache

# 確保目錄具有正確的權限
RUN chmod -R 777 /app/matplotlib_config /app/transformers_cache

# 安裝 tzdata 套件並設定時區為 Asia/Taipei
RUN apt update -y && apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
ENV TZ=Asia/Taipei

# 安裝基本套件
# libturbojpeg -> 用於 JPEG 圖像的庫
# exiftool -> 用於讀取圖像 EXIF 資訊
# ffmpeg -> 處理音頻和視頻的工具
# poppler-utils -> 用於 PDF 轉換成影像
# libpng-dev -> 用於處理 PNG 圖像的庫
# libtiff5-dev -> 用於處理 TIFF 圖像的庫
# libjpeg8-dev -> 用於處理 JPEG 圖像的庫
# libopenjp2-7-dev -> 用於處理 JPEG 2000 圖像的庫
# zlib1g-dev -> 用於壓縮和解壓縮的庫
# libfreetype6-dev -> 用於處理 TrueType 和 OpenType 字體的庫
# liblcms2-dev -> 用於處理色彩管理系統的庫
# libwebp-dev -> 用於處理 WebP 圖像的庫
# tcl8.6-dev -> GUI 工具包
# tk8.6-dev -> GUI 工具包
# python3-tk -> GUI 工具包
# libharfbuzz-dev -> 用於處理 Unicode 文本的庫
# libfribidi-dev -> 用於處理 Unicode 文本的庫
# libxcb1-dev -> X 協議 C-language Binding庫
# libfftw3-dev -> 用於處理快速傅立葉轉換的庫
RUN apt update -y && apt upgrade -y && apt install -y git \
    libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U setuptools pip wheel

# For training
RUN python -m pip install -U tqdm colored ipython tabulate \
    tensorboard scikit-learn fire albumentations Pillow>=10.0.0 fitsne

# 安裝 docsaidkit
COPY . /usr/local/DocsaidKit
RUN cd /usr/local/DocsaidKit && python setup.py bdist_wheel && \
    python -m pip install $(ls dist/*.whl | sed 's/$/[torch]/') && \
    rm -rf /usr/local/DocsaidKit

# Fixed 4.8.0.76 import error
RUN python -m pip install opencv-python==4.8.0.74

# Preload data
RUN python -c "import docsaidkit"

WORKDIR /code

CMD ["bash"]
