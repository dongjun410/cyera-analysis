import argparse
import sys
import yaml
from cyera_bench.orchestrator import BenchmarkOrchestrator
from cyera_bench.reporter import Reporter

def main():
    parser = argparse.ArgumentParser(description="Cyera FLAN-T5 NER/PII Benchmark")
    parser.add_argument("--config", "-c", type=str, help="Path to experiment YAML config file")
    parser.add_argument("--compare", nargs="+", type=str, default=None, help="Compare multiple result JSON files")
    parser.add_argument("--defaults", type=str, default=None, help="Path to defaults YAML file to merge")

    args = parser.parse_args()

    if args.compare:
        reporter = Reporter()
        reporter.compare(args.compare)
        return

    if not args.config:
        print("Usage: python -m cyera_bench --config config/experiments/<name>.yaml")
        print("   or: python -m cyera_bench --compare results/*.json")
        sys.exit(1)

    config = _load_config(args.config, args.defaults)
    orch = BenchmarkOrchestrator(config)
    result = orch.run()
    return result

def _load_config(config_path: str, defaults_path: str | None = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if defaults_path:
        with open(defaults_path, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f)
        config = _merge_configs(defaults, config)

    return config

def _merge_configs(defaults: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            defaults[key] = _merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults

if __name__ == "__main__":
    main()
