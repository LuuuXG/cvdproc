from pathlib import Path
import hipsta
import os

def main():
    hipsta_dir = Path(os.path.dirname(hipsta.__file__))
    path = hipsta_dir / "computeCubeParam.py"

    if not path.exists():
        raise FileNotFoundError(f"computeCubeParam.py not found: {path}")

    original_text = path.read_text()
    text = original_text

    marker = "from sklearn.decomposition import PCA\n"
    helper = marker + "\n\ndef _mode_1d(x):\n    return np.atleast_1d(st.mode(x, keepdims=True)[0])\n"

    if "def _mode_1d(x):" not in text:
        if marker not in text:
            raise RuntimeError("Import marker not found.")
        text = text.replace(marker, helper, 1)

    text = text.replace(
        "np.concatenate((i4c, st.mode(i4c[t4c[i, :]])[0]))",
        "np.concatenate((i4c, _mode_1d(i4c[t4c[i, :]])))"
    )

    text = text.replace(
        "np.concatenate((k4c, st.mode(k4c[t4c[i, :]])[0]))",
        "np.concatenate((k4c, _mode_1d(k4c[t4c[i, :]])))"
    )

    if text == original_text:
        print("No changes were needed.")
        print(f"Target file: {path}")
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(original_text)

    path.write_text(text)

    print(f"Hipsta directory: {hipsta_dir}")
    print(f"Patched file: {path}")
    print(f"Backup file: {backup_path}")

if __name__ == "__main__":
    main()