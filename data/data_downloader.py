import os
import time
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import yaml

# ================= 配置区域 =================
# 1. 强制使用国内镜像 (解决 Featurize 无法连接 HF 的问题)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

with open("./configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

REPO_ID = "Skywork/SkyPile-150B"
LOCAL_DIR = config["pretrain_data_dir"]  # 你的目标路径
START_INDEX = 0
END_INDEX = 40  
# ===========================================

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"正在连接 Hugging Face 镜像站 ({os.environ['HF_ENDPOINT']})...")
print(f"目标仓库: {REPO_ID}")

try:
    # 获取文件列表
    print("正在获取文件列表（可能需要几秒钟）...")
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # 调试：打印前5个文件，看看长什么样
    print(f"仓库文件总数: {len(all_files)}")
    if len(all_files) > 0:
        print(f"文件示例: {all_files[:5]}")
    else:
        print("警告：未找到任何文件，请检查网络或仓库ID。")
        exit()

    # 过滤出数据文件 (.jsonl.zst)
    # SkyPile 的文件通常在根目录，但也可能变动，我们打印出来确认了
    data_files = [f for f in all_files if f.endswith(".jsonl")]
    data_files.sort()

    print(f"找到 .jsonl 数据文件: {len(data_files)} 个")

    if not data_files:
        print("错误：没有找到 .jsonl 结尾的文件。")
        exit()

    # 截取需要的部分
    target_files = data_files[START_INDEX:END_INDEX]
    print(f"准备下载索引 {START_INDEX} 到 {END_INDEX} 的文件 (共 {len(target_files)} 个)")

    for file_path in tqdm(target_files, desc="Downloading"):
        # 检查文件是否已存在，避免重复下载
        local_file_path = os.path.join(LOCAL_DIR, file_path)
        if os.path.exists(local_file_path):
            print(f"文件已存在，跳过: {file_path}")
            continue
            
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True # 支持断点续传
            )
        except Exception as e:
            print(f"\n下载 {file_path} 失败: {e}")
            # 失败重试一次
            print("尝试重试...")
            time.sleep(2)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True
            )

    print("\n全部下载任务完成！")

except Exception as e:
    print(f"\n发生严重错误: {e}")
    print("建议：如果显示 Connection Error，请检查 Featurize 的网络设置或稍后再试。")
