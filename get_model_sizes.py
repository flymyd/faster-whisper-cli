import huggingface_hub
import requests
_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
def get_model_bin_total_size(repo_id, token=None):
    """
    统计指定仓库内所有名为 model.bin 及其分片（model.bin.*, 例如 model.bin.1）的文件大小总和。

    Args:
        repo_id (str): 仓库 ID（例如 'Systran/faster-whisper-large-v3'）。
        token (str, optional): Hugging Face 访问令牌。

    Returns:
        int | str: 以字节为单位的总大小；或错误描述字符串。
    """
    def _is_model_bin(path: str) -> bool:
        filename = path.split("/")[-1]
        return filename == "model.bin" or filename.startswith("model.bin.")

    try:
        api = huggingface_hub.HfApi(token=token)

        # 1) 找到所有 model.bin* 路径
        model_bin_paths = []
        try:
            paths = api.list_repo_files(repo_id=repo_id, repo_type="model")
            model_bin_paths = [p for p in paths if _is_model_bin(p)]
        except Exception:
            model_bin_paths = []

        total_size = 0

        # 2) 若找到路径，优先用批量接口或逐个接口拿大小
        if model_bin_paths:
            # 2.1) get_paths_info（新接口，若可用）
            try:
                get_paths_info = getattr(api, "get_paths_info", None)
                if callable(get_paths_info):
                    infos = api.get_paths_info(repo_id=repo_id, paths=model_bin_paths, repo_type="model")
                    for info in infos:
                        size = getattr(info, "size", None)
                        if size is not None:
                            total_size += size
                # 如果该接口不可用或返回的大小均为 None，则继续下一步
            except Exception:
                pass

            # 2.2) 逐个 file_info 获取大小
            if total_size == 0:
                for path in model_bin_paths:
                    try:
                        info = api.file_info(repo_id=repo_id, path=path, repo_type="model")
                        size = getattr(info, "size", None)
                        if size is not None:
                            total_size += size
                    except Exception:
                        # 2.3) 使用 HEAD 获取 content-length 作为兜底
                        try:
                            url = huggingface_hub.hf_hub_url(repo_id=repo_id, filename=path, repo_type="model")
                            headers = {}
                            if token:
                                headers["authorization"] = f"Bearer {token}"
                            resp = requests.head(url, allow_redirects=True, headers=headers, timeout=30)
                            if resp.ok and "content-length" in resp.headers:
                                total_size += int(resp.headers["content-length"])
                        except Exception:
                            pass

        # 3) 如果仍为 0，则退回到通过 model_info 统计（model.bin* 或整个仓库大小）
        if total_size == 0:
            try:
                model_info = huggingface_hub.model_info(repo_id, token=token)
                # 先尝试仅统计 model.bin*
                model_bin_total = 0
                for f in getattr(model_info, "siblings", []) or []:
                    path = getattr(f, "rfilename", getattr(f, "path", ""))
                    size = getattr(f, "size", None)
                    if size is not None and _is_model_bin(path):
                        model_bin_total += size
                if model_bin_total > 0:
                    return model_bin_total

                # 若依旧拿不到，则统计整个仓库大小（用户也接受）
                repo_total = sum((getattr(f, "size", 0) or 0) for f in getattr(model_info, "siblings", []) or [])
                return repo_total
            except Exception:
                pass

        return total_size
    except huggingface_hub.utils.RepositoryNotFoundError:
        return f"错误：未找到仓库 {repo_id}"
    except Exception as e:
        return f"获取 {repo_id} 的 model.bin 大小时出错: {e}"

if __name__ == "__main__":
    # Get unique repository IDs from the _MODELS dictionary to avoid duplicate checks
    unique_repo_ids = sorted(list(set(_MODELS.values())))

    print("仓库 model.bin 文件总大小:")
    print("=" * 40)

    for repo_id in unique_repo_ids:
        size_bytes = get_model_bin_total_size(repo_id)

        if isinstance(size_bytes, int):
            if size_bytes < (1024 ** 3):
                size_mb = size_bytes / (1024 ** 2)
                print(f"{repo_id:<50} {size_mb:>6.2f} MB")
            else:
                size_gb = size_bytes / (1024 ** 3)
                print(f"{repo_id:<50} {size_gb:>6.2f} GB")
        else:
            # Print error message if size calculation failed
            print(f"{repo_id:<50} {size_bytes}")

    print("=" * 40)
