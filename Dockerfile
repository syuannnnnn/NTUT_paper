# For Windows
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# 安裝 Miniconda
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop';"]

RUN Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile C:\\Miniconda3.exe ; \
    Start-Process C:\\Miniconda3.exe -ArgumentList '/InstallationType=JustMe', '/AddToPath=1', '/RegisterPython=0', '/S', '/D=C:\\Miniconda3' -NoNewWindow -Wait ; \
    Remove-Item C:\\Miniconda3.exe -Force

# 設定環境變數
ENV PATH C:\\Miniconda3;C:\\Miniconda3\\Scripts;C:\\Miniconda3\\Library\\bin;$PATH

# 複製 environment.yml
WORKDIR /app
COPY environment.yml .

# 建立 conda 環境
RUN conda env create -f environment.yml

# 預設啟動環境
SHELL ["cmd", "/S", "/C"]
RUN echo conda activate tripo > C:\\init_env.bat

ENTRYPOINT ["cmd", "/K", "C:\\init_env.bat"]


# For Linux
# FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# RUN apt-get update && apt-get install -y wget git bash && rm -rf /var/lib/apt/lists/*

# # 安裝 Miniconda
# ENV CONDA_DIR=/opt/conda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
#     && bash miniconda.sh -b -p $CONDA_DIR \
#     && rm miniconda.sh
# ENV PATH=$CONDA_DIR/bin:$PATH

# # 複製 environment.yml
# COPY environment.yml /tmp/environment.yml

# # 建立 conda 環境
# RUN conda env create -f /tmp/environment.yml

# # 預設使用你的虛擬環境
# SHELL ["conda", "run", "-n", "tripo", "/bin/bash", "-c"]

# # 設定容器內的工作資料夾
# WORKDIR /app

# # 複製你的專案程式碼
# COPY . /app

# CMD ["python", "main.py"]
