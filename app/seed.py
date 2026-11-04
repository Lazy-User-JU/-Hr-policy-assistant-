from pathlib import Path

from app import store
from app.ingest import index_document

POLICIES = Path(__file__).parent.parent / "policies"


def main() -> None:
    store.init()
    for path in sorted(POLICIES.iterdir()):
        result = index_document(path.name, path.read_bytes())
        print(f"{result.document}: {result.status}, version {result.version}, {result.chunks} chunks")


if __name__ == "__main__":
    main()
