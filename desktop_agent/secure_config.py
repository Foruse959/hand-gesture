from __future__ import annotations

import argparse
import json
from pathlib import Path

from cryptography.fernet import Fernet


def _resolve_key(raw_or_path: str) -> str:
    candidate = Path(raw_or_path)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return raw_or_path.strip()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def generate_key(out_path: str | None) -> None:
    key = Fernet.generate_key().decode("utf-8")
    if out_path:
        _write_text(Path(out_path), key + "\n")
        print(f"Wrote key file: {out_path}")
    print("Fernet key:")
    print(key)


def encrypt_json(input_path: str, output_path: str, key_or_path: str) -> None:
    source = Path(input_path)
    raw = source.read_text(encoding="utf-8")
    json.loads(raw)

    key = _resolve_key(key_or_path)
    token = Fernet(key.encode("utf-8")).encrypt(raw.encode("utf-8"))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(token)
    print(f"Encrypted config written: {output_path}")


def decrypt_json(input_path: str, output_path: str, key_or_path: str) -> None:
    source = Path(input_path)
    token = source.read_bytes()

    key = _resolve_key(key_or_path)
    plaintext = Fernet(key.encode("utf-8")).decrypt(token)

    parsed = json.loads(plaintext.decode("utf-8"))
    pretty = json.dumps(parsed, indent=2)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(pretty + "\n", encoding="utf-8")
    print(f"Decrypted config written: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encrypt/decrypt desktop_agent config with Fernet")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-key", help="Generate a Fernet key")
    gen.add_argument("--out", default="", help="Optional output path for key file")

    enc = sub.add_parser("encrypt", help="Encrypt a JSON config file")
    enc.add_argument("--in", dest="input_path", required=True, help="Path to plaintext JSON config")
    enc.add_argument("--out", dest="output_path", required=True, help="Path to encrypted config")
    enc.add_argument("--key", required=True, help="Fernet key value or key file path")

    dec = sub.add_parser("decrypt", help="Decrypt a config file")
    dec.add_argument("--in", dest="input_path", required=True, help="Path to encrypted config")
    dec.add_argument("--out", dest="output_path", required=True, help="Path to plaintext JSON config")
    dec.add_argument("--key", required=True, help="Fernet key value or key file path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "generate-key":
        generate_key(args.out or None)
    elif args.command == "encrypt":
        encrypt_json(args.input_path, args.output_path, args.key)
    elif args.command == "decrypt":
        decrypt_json(args.input_path, args.output_path, args.key)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
