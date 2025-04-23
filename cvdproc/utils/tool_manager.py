import argparse
import subprocess
import sys
import os

def list_tools(base_dir):
    """列出所有可用工具"""
    print("Available tools:")
    for folder in ["python", "bash", "matlab"]:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            for script in os.listdir(folder_path):
                print(f" - {folder}/{script}")

def run_python_tool(script_path, args):
    """运行 Python 工具"""
    command = [sys.executable, script_path] + args
    subprocess.run(command, check=True)

def run_bash_tool(script_path, args):
    """运行 Bash 脚本"""
    command = ["bash", script_path] + args
    subprocess.run(command, check=True)

def run_matlab_tool(script_path, args):
    """运行 MATLAB 脚本"""
    command = ["matlab", "-batch", f"run('{script_path}');"] + args
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Tool manager for cerebrovascular disease utilities.")
    parser.add_argument("tool", help="Tool to run in the format 'folder/tool_name'. Use 'list' to show available tools.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional arguments to pass to the tool.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.tool == "list":
        list_tools(base_dir)
        return

    # 解析工具路径
    try:
        folder, tool_name = args.tool.split("/")
        script_path = os.path.join(base_dir, folder, tool_name)
    except ValueError:
        print("Tool name must be in the format 'folder/tool_name'. Use 'list' to see available tools.")
        sys.exit(1)

    if not os.path.exists(script_path):
        print(f"Tool '{args.tool}' not found.")
        sys.exit(1)

    # 根据文件类型运行工具
    if folder == "python" and tool_name.endswith(".py"):
        run_python_tool(script_path, args.args)
    elif folder == "bash" and tool_name.endswith(".sh"):
        run_bash_tool(script_path, args.args)
    elif folder == "matlab" and tool_name.endswith(".m"):
        run_matlab_tool(script_path, args.args)
    else:
        print(f"Unsupported tool format for '{args.tool}'.")

if __name__ == "__main__":
    main()
