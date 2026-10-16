"""Local JSON validation only; help and imports have no device side effects."""
import argparse
from pathlib import Path
from .protocols import Event, ProtocolError


def main():
    parser = argparse.ArgumentParser(description="Validate one public BCI/VLA JSON event offline.")
    parser.add_argument("path", nargs="?", type=Path, help="event JSON file; no network requests")
    args = parser.parse_args()
    if args.path is None:
        parser.print_help()
        return 0
    try:
        event = Event.parse(args.path.read_text(encoding="utf-8-sig"))
    except (ProtocolError, OSError) as exc:
        parser.exit(2, f"Invalid: {exc if isinstance(exc, ProtocolError) else type(exc).__name__}\n")
    print(f"VALID {event.data['kind']} {event.data['mode']} (schema only; not authorized execution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
