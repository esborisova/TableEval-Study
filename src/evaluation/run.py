import argparse
from typing import Union
import ast


from evaluator import Evaluator
from models import HFModel
from utils import save_results


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="",
        help="Name of model e.g. `occiglot/occiglot-7b-eu5`",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="{}",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=0,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default="output",
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `0` "
            "(for backward compatibility).\n"
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="saves the samples together with the scores",
    )
    parser.add_argument(
        "--log_logits",
        action="store_true",
        help="saves the samples together with the scores. If log_logits=True the samples will also be saved.",
    )
    parser.add_argument(
        "--multi_modal",
        "-mm",
        action="store_true",
        help="if running multi modal LLMs. default: False",
    )
    parser.add_argument(
        "--image_special_token",
        type=str,
        default="<image>",
        help="Changing the special token for the multi_modal LLMs. The default is <image>",
    )
    return parser


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        parser = setup_parser()
        args = parser.parse_args()
    if args.log_logits and not args.log_samples:
        args.log_samples = True
    model = HFModel(
        args.model_name,
        ast.literal_eval(args.model_args),
        batch_size=args.batch_size,
        random_seed=args.seed,
        device=args.device,
        multi_modal=args.multi_modal,
        special_token_for_image=args.image_special_token,
    )

    eval = Evaluator(
        model=model,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        random_seed=args.seed,
        log_samples=args.log_samples,
    )

    results = eval.simple_eval()

    save_results(args.output_path, results, model_name=model.get_model_info(), log_logits=args.log_logits)


if __name__ == "__main__":
    cli_evaluate()
