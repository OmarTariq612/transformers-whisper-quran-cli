import click
from clicktypes import SORAH_RANGE
from merge import merge as m


@click.group()
def main():
    pass


@main.command(help="generate csv files containing WER for the given input and model")
@click.argument(
    "text-csv-path", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "audio-path", type=click.Path(file_okay=False, exists=True, executable=True)
)
@click.option(
    "--model",
    type=str,
    help="multilingual model used for transcribing",
)
@click.option("--device", type=str, default="cpu", help="device used to load the model")
@click.option("--sorah-range", default="1:114", type=SORAH_RANGE)
@click.option(
    "--out-prefix",
    type=click.STRING,
)
@click.option(
    "--log-level",
    default="normal",
    type=click.Choice(("normal", "verbose")),
    help="determine the logging level (default: normal)",
)
@click.option(
    "-o",
    default=".",
    type=click.Path(file_okay=False, executable=True),
    help="output directory",
)
@click.option(
    "--batch-size",
    "-b",
    default=8,
    type=click.INT,
    help="batch size used for transcribing (default: 8)",
)
def generate(
    text_csv_path: str,
    audio_path: str,
    model: str,
    device: str,
    sorah_range: tuple[int, int],
    out_prefix: str,
    log_level: str,
    o: str,
    batch_size: int,
):
    from transcribe import transcribe

    transcribe(
        audio_path=audio_path,
        text_csv_path=text_csv_path,
        model_str=model,
        device=device,
        from_sorah=sorah_range[0],
        to_sorah=sorah_range[1],
        output_dir=o,
        out_prefix=out_prefix,
        log_level=log_level,
        batch_size=batch_size,
    )


# @main.command(help="merge multiple csv files into one containing WER as a total")
# @click.argument(
#     "src",
#     type=click.Path(exists=True, dir_okay=False, readable=True),
#     nargs=-1,
#     required=True,
# )
# @click.option("--out-prefix", default="merged", type=click.STRING)
# @click.option(
#     "-o", default=".", type=click.Path(exists=True, file_okay=False, executable=True)
# )
# def merge(src: tuple[str, ...], out_prefix: str, o: str):
#     m(srcs=src, out_prefix=out_prefix, output_dir=o)


if __name__ == "__main__":
    main()
