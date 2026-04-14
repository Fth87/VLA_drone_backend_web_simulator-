from pathlib import Path


def main() -> None:
    path = Path("/root/openpi/src/openpi/models/model.py")
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"No-op patch helper: OpenPI model loader is now maintained in-source at {path}")


if __name__ == '__main__':
    main()
