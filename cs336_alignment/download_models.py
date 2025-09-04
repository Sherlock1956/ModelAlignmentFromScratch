from modelscope import snapshot_download

def download_qwen25_math():
    # 模型仓库名称（在 ModelScope 上的 ID）
    repo_id = "qwen/Qwen2.5-0.5B"

    # 本地保存目录
    local_dir = "/root/autodl-tmp/lyx/CS336-assignment5/models/Qwen2.5-0.5B"

    print(f"Downloading {repo_id} from ModelScope to {local_dir} ...")

    # 下载模型
    snapshot_download(
        model_id=repo_id,
        cache_dir=local_dir,
        revision="master"  # 可以指定分支/版本
    )

    print("Download completed!")

if __name__ == "__main__":
    download_qwen25_math()