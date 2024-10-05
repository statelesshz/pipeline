from pathlib import Path

def list_files(directory):
    """
    递归遍历目录，返回所有文件的路径
    :param directory: 要遍历的目录
    :return: 包含所有文件路径的列表
    """
    path = Path(directory)
    return [str(entry) for entry in path.rglob("*") if entry.is_file()]

if __name__ == "__main__":
    res1 = list_files("/home/lynn/github/qwen2.5-0.5b-instruct")
    res2 = list_files("/home/lynn/github/sd1-4")
    for res in res1:
        if res.startswith("diffusion_pytorch_model"):
            print("diffusers")
    print("----")
    for res in res2:
        if res.startswith("diffusion_pytorch_model"):
            print("diffusers")
