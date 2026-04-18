from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import default_proxy_keys, default_teacher_key, load_model_zoo, resolve_model_spec
from .data import calibration_pairs_from_examples, build_prompt, load_gsm8k_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="FROST modular CLI")
    parser.add_argument("--model-zoo", default=str(Path(__file__).resolve().parents[1] / "model_zoo.json"), help="Path to model_zoo.json")
    parser.add_argument("--teacher-key", default=None, help="Teacher key from the model zoo")
    parser.add_argument("--proxy-keys", nargs="*", default=None, help="Proxy keys from the model zoo")
    parser.add_argument("--dtype", default=None, help="Override dtype for all models, e.g. float16, bfloat16, float32")
    parser.add_argument("--device-map", default=None, help="Override device map for all models, e.g. auto or cpu")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--shortlist-k", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--sample", action="store_true", help="Sample from the shortlist instead of greedy decoding")
    parser.add_argument("--kfac-rank", type=int, default=64)
    parser.add_argument("--kfac-damping", type=float, default=1e-3)
    parser.add_argument("--kfac-layer-limit", type=int, default=16)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--output-json", default=None, help="Where to write results JSON for decode/evaluate")
    parser.add_argument("--plot-dir", default="plots", help="Directory for generated plots")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decode = subparsers.add_parser("decode", help="Run one defended decode")
    decode.add_argument("--prompt", default=None, help="Prompt text")
    decode.add_argument("--prompt-file", default=None, help="Read prompt text from file")

    evaluate = subparsers.add_parser("evaluate", help="Run GSM8K evaluation")
    evaluate.add_argument("--eval-limit", type=int, default=8, help="How many GSM8K rows to evaluate")
    evaluate.add_argument("--beta-sweep", nargs="*", type=float, default=None, help="Optional list of beta values for tradeoff plots")

    plot = subparsers.add_parser("plot", help="Plot an existing result JSON")
    plot.add_argument("--results-json", default=None, help="Path to a results JSON file")

    return parser


def _resolve_key_list(zoo, keys, default_keys, role):
    if keys is None or len(keys) == 0:
        keys = default_keys
    if not keys:
        raise ValueError(f"No {role} keys were provided and the model zoo has no defaults.")
    return keys


def _load_models(args, zoo):
    from .kfac import KFACConfig, build_proxy_state, calibrate_proxy, clear_proxy_hooks, register_kfac_hooks
    from .models import load_model_bundle

    teacher_key = args.teacher_key or default_teacher_key(zoo)
    if teacher_key is None:
        raise ValueError("No teacher key provided and the model zoo has no default teacher_key.")
    proxy_keys = _resolve_key_list(zoo, args.proxy_keys, default_proxy_keys(zoo), "proxy")

    teacher_spec = resolve_model_spec(zoo, teacher_key, expected_role="teacher")
    teacher_tokenizer, teacher_model = load_model_bundle(
        teacher_spec,
        dtype_override=args.dtype,
        device_map_override=args.device_map,
    )

    proxy_states = []
    kfac_config = KFACConfig(rank=args.kfac_rank, damping=args.kfac_damping, layer_limit=args.kfac_layer_limit)
    examples_for_calibration = load_gsm8k_dataset(limit=max(args.calibration_samples, 1))
    calibration_pairs = calibration_pairs_from_examples(examples_for_calibration, limit=args.calibration_samples)

    for proxy_index, proxy_key in enumerate(proxy_keys):
        proxy_spec = resolve_model_spec(zoo, proxy_key, expected_role="proxy")
        proxy_tokenizer, proxy_model = load_model_bundle(
            proxy_spec,
            dtype_override=args.dtype,
            device_map_override=args.device_map,
        )
        proxy = build_proxy_state(f"proxy_{proxy_index}:{proxy_key}", proxy_model, proxy_tokenizer, config=kfac_config)
        register_kfac_hooks(proxy)
        calibrate_proxy(proxy, calibration_pairs, max_samples=args.calibration_samples, max_length=args.max_sequence_length)
        clear_proxy_hooks(proxy)
        proxy_states.append(proxy)
        print(f"Calibrated {proxy.name} with {len(proxy.layers)} K-FAC layers")

    return teacher_model, teacher_tokenizer, proxy_states


def _decode_prompt(args, teacher_model, teacher_tokenizer, proxy_states):
    from .decoding import DecodeConfig, frost_generate, teacher_generate

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = input("Prompt: ").strip()
    decode_config = DecodeConfig(
        max_sequence_length=args.max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        shortlist_k=args.shortlist_k,
        beta=args.beta,
        sample=args.sample,
        lambda_geom=1.0,
        lambda_struct=0.0,
        lambda_prov=0.0,
    )
    teacher_out, teacher_stats = teacher_generate(
        teacher_model,
        teacher_tokenizer,
        prompt,
        decode_config,
        sample=args.sample,
        return_stats=True,
    )
    frost_out, frost_stats = frost_generate(
        teacher_model,
        teacher_tokenizer,
        proxy_states,
        prompt,
        decode_config,
        sample=args.sample,
        return_stats=True,
    )
    payload = {
        "prompt": prompt,
        "teacher": teacher_out,
        "frost": frost_out,
        "teacher_stats": teacher_stats,
        "frost_stats": frost_stats,
        "beta": args.beta,
        "shortlist_k": args.shortlist_k,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _evaluate(args, teacher_model, teacher_tokenizer, proxy_states):
    from .decoding import DecodeConfig
    from .evaluation import run_gsm8k_evaluation, save_evaluation_results
    from .plots import plot_results

    examples = load_gsm8k_dataset(limit=args.eval_limit)
    beta_values = args.beta_sweep if args.beta_sweep else [args.beta]
    decode_config = DecodeConfig(
        max_sequence_length=args.max_sequence_length,
        max_new_tokens=args.max_new_tokens,
        shortlist_k=args.shortlist_k,
        beta=args.beta,
        sample=args.sample,
        lambda_geom=1.0,
        lambda_struct=0.0,
        lambda_prov=0.0,
    )
    rows, summary = run_gsm8k_evaluation(
        teacher_model,
        teacher_tokenizer,
        proxy_states,
        examples,
        decode_config,
        eval_limit=args.eval_limit,
        beta_values=beta_values,
        sample=args.sample,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    output_json = args.output_json or "gsm8k_frost_results.json"
    save_evaluation_results(rows, output_json)
    print(f"Saved results to {output_json}")
    plot_results(rows, output_dir=args.plot_dir)
    return 0


def _plot(args):
    from .plots import plot_results

    results_json = args.results_json or args.output_json or "gsm8k_frost_results.json"
    rows = json.loads(Path(results_json).read_text(encoding="utf-8"))
    plot_results(rows, output_dir=args.plot_dir)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "decode":
        zoo = load_model_zoo(args.model_zoo)
        teacher_model, teacher_tokenizer, proxy_states = _load_models(args, zoo)
        return _decode_prompt(args, teacher_model, teacher_tokenizer, proxy_states)
    if args.command == "evaluate":
        zoo = load_model_zoo(args.model_zoo)
        teacher_model, teacher_tokenizer, proxy_states = _load_models(args, zoo)
        return _evaluate(args, teacher_model, teacher_tokenizer, proxy_states)
    if args.command == "plot":
        return _plot(args)
    raise ValueError(f"Unknown command: {args.command}")
