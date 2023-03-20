#!/usr/bin/env python3
import argparse
import collections
import functools
import itertools
import logging
import pathlib
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Union

import jsonlines


DESCRIPTION = """Reformat a dataset to make it suitable for fine tuning

According to the OpenAI API documentation, fine tuning data must be in the form
of a JSONL file where each line look like this:

    {"prompt": "Knock, knock", "completion: "Who's there?"}

For more information, see
https://platform.openai.com/docs/guides/fine-tuning/prepare-training-data

This script takes a JSONL dataset and transforms it to match the above format.
The contents of the prompt and completion are determined by templates within
this script. It uses Python's `str.format()` method to populate the template
with the contents of each entry. Read more about the syntax here:
https://docs.python.org/3/library/string.html#formatstrings

The template should use the keys in the original dataset. If a key is not
present in an entry, the empty string will be inserted instead.
"""

EPILOG = r"""EXAMPLES:

# Minimal example to format original dataset for fine tuning
python format_for_fine_tuning.py -i alignment_texts.jsonl -o fine-tuning.jsonl

# Only include entries from "lesswong" and "alignment forum"
python format_for_fine_tuning.py -i alignment_texts.jsonl -o fine-tuning.jsonl \
    --sources lesswrong "alignment forum"

# Or if the original dataset is already split by source, you can specify which
# files to use as input
python format_for_fine_tuning.py -i arxiv.jsonl lesswrong.jsonl -o fine-tuning.jsonl
"""

# Template for formatting the prompt.
# OpenAI recommends the prompt end with a stop sequence (e.g. "\n\n###\n\n") to divide
# it from the completion.
PROMPT_TEMPLATE = """
{title}

by {authors}
Tags: {tags}
Votes: {votes}
Karma: {score}

{text}

%%%

"""

# Template for formatting the completion.
# OpenAI recommends the completion start with a whitespace character and end
# with a stop sequence (e.g. "\n\n###\n\n").
COMPLETION_TEMPLATE = """ {comments}

%END%

"""

# Template for formatting forum comments, like those from LessWrong.
COMMENT_TEMPLATE = """Comment by {authors}
Votes: {votes}
Karma: {score}

{text}"""


# The FIELD_FORMATTERS dictionary defines custom formatters for specific fields in a
# larger data entry. These are applied before the prompt and completion are created.
# They offer more control over how individual fields, like the main text, are formatted
# in the final output.

# Each formatter is matched to the corresponding field using its key. A formatter can be
# a string, in which case it will be interpreted as a template and applied using
# format_entry(). Or a formatter can be a callable that takes the field value and
# returns a string.
FIELD_FORMATTERS: Mapping[str, Union[str, Callable[[str], str]]] = {
    # Format the comment field according to the template above
    "comments": COMMENT_TEMPLATE,
    # Replace minus sign (Unicode code point U+2212) with plain ASCII hyphen (U+002D)
    "score": lambda s: s.replace("−", "-"),
    # Replace instances of the separator within the text
    "text": lambda s: s.replace("%%%", "---"),
}


def format_fields(
    entry: MutableMapping[str, Any],
    formatters: Mapping[str, Union[str, Callable[[Any], str]]] = FIELD_FORMATTERS,
) -> None:
    """Format fields according to their own templates

    This function modifies the input mapping in-place.

    As an example, consider this input entry:
        {
            "title": "A Proposal",
            "comments": [
                {
                    "author": "Foo Bar",
                    "text": "Nice idea!"
                }
            ]
        }

    With the appropriate templates, the `comments` field could be reformatted
    as follows:
        {
            "title": "A Proposal",
            "comments": "Foo Bar\n\nNice idea!"
        }
    """
    for key, value in entry.items():
        if key not in formatters:
            continue
        format_this_field: Callable[[Any], str]
        if callable(formatters[key]):
            format_this_field = formatters[key]  # type: ignore[assignment]
        else:
            format_this_field = functools.partial(
                format_entry, template=formatters[key]
            )
        if isinstance(value, collections.abc.MutableSequence) and not isinstance(
            value, str
        ):
            for i, elem in enumerate(value):
                # TODO: What about lists of lists or lists of ints?
                if isinstance(elem, collections.abc.MutableMapping):
                    value[i] = format_this_field(elem)
            entry[key] = "\n\n\n".join(value)
        else:
            entry[key] = format_this_field(value)


def format_entry(entry: Mapping[str, Any], template: str) -> str:
    """Format an entry according to the given template"""
    # Fill in missing values with the empty string
    entry_with_default = collections.defaultdict(str, entry)
    format_fields(entry_with_default)
    return template.format_map(entry_with_default)


def format_prompt(entry: Mapping[str, Any]) -> str:
    return format_entry(entry, PROMPT_TEMPLATE)


def format_completion(entry: Mapping[str, Any]) -> str:
    return format_entry(entry, COMPLETION_TEMPLATE)


def prepare_fine_tuning_entries(
    input_paths: Sequence[pathlib.Path],
    output_path: pathlib.Path,
    sources: Optional[Sequence[str]] = None,
) -> None:
    lines_read = 0
    lines_written = 0
    with jsonlines.open(output_path, "w", compact=True) as writer:
        for i, input_path in enumerate(input_paths):
            input_parse_errors = 0
            logging.info(
                f"Processing input file {i + 1}/{len(input_paths)}: {input_path}"
            )
            with open(input_path) as input_file:
                reader = jsonlines.Reader(input_file)
                for line_number in itertools.count(1):
                    try:
                        input_entry = reader.read(type=dict, skip_empty=True)
                    except EOFError:
                        # Reached end of file, continue to the next one
                        break
                    except jsonlines.InvalidLineError as err:
                        logging.debug(f"Skipping line {line_number} due to {err!r}")
                        input_parse_errors += 1
                    else:
                        lines_read += 1
                        if (
                            sources is not None
                            and input_entry.get("source") not in sources
                        ):
                            # Skip this entry because it's source
                            # isn't one we want to include
                            continue
                        lines_written += write_entry(input_entry, writer)

            if input_parse_errors > 0:
                logging.warning(f"Skipped {input_parse_errors} malformed lines")

    logging.info(f"Processed {lines_read} lines, wrote {lines_written} lines")


def write_entry(input_entry: Mapping[str, Any], writer: jsonlines.Writer) -> int:
    lines_written = 0
    # Make a separate output entry for each top-level comment
    for comment_index, comment in enumerate(input_entry.get("comments", [])):
        try:
            entry_with_one_comment = dict(input_entry)
            entry_with_one_comment["comments"] = [comment]
            output_entry = {
                "prompt": format_prompt(entry_with_one_comment),
                "completion": format_completion(entry_with_one_comment),
            }
        except (AttributeError, KeyError, ValueError) as err:
            logging.debug(f"Skipping comment {comment_index} due to {err!r}")
        else:
            writer.write(output_entry)
            lines_written += 1
    return lines_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        type=pathlib.Path,
        help="Path(s) to JSONL dataset(s) to reformat for fine tuning",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="Path to file where JSONL fine tuning data will be saved",
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        help="""Filter entries from input dataset to only include these sources.
        By default, include all sources.""",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Assume a reply of 'yes' for all prompts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output extra info to help with debugging.",
    )
    args = parser.parse_args()

    logging_level = logging.INFO
    if args.verbose:
        logging_level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging_level)

    if args.output.exists() and not args.yes:
        logging.warning(f"Output file '{args.output}' exists.")
        reply = ""
        while reply not in ("y", "yes", "n", "no"):
            reply = input("Overwrite? (y/n): ").lower()
        if reply not in ("y", "yes"):
            logging.info("Exiting")
            return

    prepare_fine_tuning_entries(
        input_paths=args.input,
        output_path=args.output,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()
