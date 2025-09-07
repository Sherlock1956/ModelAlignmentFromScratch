from modelscope import snapshot_download

def download_qwen25_math():
    # 模型仓库名称（在 ModelScope 上的 ID）
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 本地保存目录
    local_dir = "/Users/lyx/Downloads/Study/projects/python/CS336-assignment5/models/Qwen2.5-0.5B-Instruct"

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