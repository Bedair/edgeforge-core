"""EdgeForge CLI — main entry point."""
import click
from rich.console import Console

console = Console()

@click.group()
@click.version_option()
def main():
    """EdgeForge — forge your models into firmware."""
    pass

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", default=None, help="Target MCU (e.g. stm32f407)")
def analyze(model_path, mcu):
    """Analyze a model — format, ops, RAM/flash estimate, board compatibility."""
    console.print(f"[bold]EdgeForge[/bold] analyzing [cyan]{model_path}[/cyan]...")
    raise NotImplementedError("Phase 1 — not yet implemented")

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", required=True, help="Target MCU profile")
@click.option("--output", "-o", default="optimized.onnx")
def optimize(model_path, mcu, output):
    """Optimize a model to fit the target MCU budget."""
    console.print(f"[bold]EdgeForge[/bold] optimizing for [cyan]{mcu}[/cyan]...")
    raise NotImplementedError("Phase 2 — not yet implemented")

@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--mcu", required=True)
@click.option("--rtos", default="none", type=click.Choice(["none", "freertos", "zephyr"]))
@click.option("--output-dir", "-o", default="edgeforge_output")
def compile(model_path, mcu, rtos, output_dir):
    """Compile a model to C/C++ for the target MCU."""
    console.print(f"[bold]EdgeForge[/bold] compiling for [cyan]{mcu}[/cyan] (rtos=[cyan]{rtos}[/cyan])...")
    raise NotImplementedError("Phase 3 — not yet implemented")

@main.command()
@click.option("--mcu", default=None)
def targets(mcu):
    """List all supported MCU targets."""
    console.print("[bold]Supported targets:[/bold]")
    raise NotImplementedError("Phase 1 — not yet implemented")

if __name__ == "__main__":
    main()
