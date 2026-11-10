import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="profilebo", description="...")
    sub = parser.add_subparsers(dest="command", required=True)
    # later: p = sub.add_parser("run", help="..."); p.add_argument("config"); p.set_defaults(func=...)
    args = parser.parse_args(argv)
    args.func(args)
