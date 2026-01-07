"""CLI for AWS troubleshooting agent."""

import asyncio
import click
import logging
import os
import sys
from dotenv import load_dotenv

from aws_troubleshooter.agent import AWSTroubleshootingAgent


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
def cli():
    """AWS Troubleshooting Agent - Diagnose AWS application issues using MCP servers."""
    pass


@cli.command()
@click.option('--service-name', '-s', required=True, help='Name of the AWS service/application to troubleshoot')
@click.option('--issue', '-i', required=True, help='Description of the issue')
@click.option('--log-level', '-l', default='INFO', help='Log level (DEBUG, INFO, WARNING, ERROR)')
def troubleshoot(service_name: str, issue: str, log_level: str):
    """Troubleshoot an AWS application issue."""
    load_dotenv()
    setup_logging(log_level)

    # Check for required environment variables
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        click.echo("Error: ANTHROPIC_API_KEY environment variable not set", err=True)
        click.echo("Please set it in .env file or export it", err=True)
        sys.exit(1)

    aws_profile = os.getenv('AWS_PROFILE', 'default')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')

    click.echo(f"\n🔍 AWS Troubleshooting Agent")
    click.echo(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    click.echo(f"Service: {service_name}")
    click.echo(f"Issue: {issue}")
    click.echo(f"AWS Profile: {aws_profile}")
    click.echo(f"AWS Region: {aws_region}")
    click.echo(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    async def run_troubleshooting():
        agent = AWSTroubleshootingAgent(api_key)

        try:
            click.echo("📡 Initializing MCP clients...")
            await agent.initialize()
            click.echo("✓ MCP clients initialized\n")

            click.echo("🔎 Starting troubleshooting workflow...")
            result = await agent.troubleshoot(service_name, issue)

            click.echo("\n" + "="*60)
            click.echo("📊 TROUBLESHOOTING RESULTS")
            click.echo("="*60 + "\n")

            click.echo(f"Service Type: {result['service_type']}\n")

            click.echo("🔍 Findings:")
            click.echo("-" * 60)
            for key, value in result['findings'].items():
                click.echo(f"\n{key}:")
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "...(truncated)"
                click.echo(f"{value_str}\n")

            click.echo("\n" + "="*60)
            click.echo("🎯 ROOT CAUSE")
            click.echo("="*60)
            click.echo(f"\n{result['root_cause']}\n")

            click.echo("="*60)
            click.echo("💡 RECOMMENDATIONS")
            click.echo("="*60)
            for i, rec in enumerate(result['recommendations'], 1):
                click.echo(f"\n{i}. {rec}")

            click.echo("\n" + "="*60 + "\n")

        except Exception as e:
            click.echo(f"\n❌ Error: {str(e)}", err=True)
            logging.exception("Troubleshooting failed")
            sys.exit(1)

        finally:
            click.echo("🧹 Cleaning up...")
            await agent.shutdown()
            click.echo("✓ Done\n")

    asyncio.run(run_troubleshooting())


@cli.command()
def version():
    """Show version information."""
    from aws_troubleshooter import __version__
    click.echo(f"AWS Troubleshooting Agent v{__version__}")


@cli.command()
def check():
    """Check if all dependencies are installed."""
    load_dotenv()
    setup_logging("ERROR")

    click.echo("🔍 Checking dependencies...\n")

    # Check Python version
    import sys
    py_version = sys.version_info
    click.echo(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Check for ANTHROPIC_API_KEY
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        click.echo(f"✓ ANTHROPIC_API_KEY is set")
    else:
        click.echo(f"✗ ANTHROPIC_API_KEY is not set", err=True)

    # Check for AWS credentials
    aws_profile = os.getenv('AWS_PROFILE', 'default')
    click.echo(f"✓ AWS_PROFILE: {aws_profile}")

    # Check if uvx is available
    import subprocess
    try:
        result = subprocess.run(['uvx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            click.echo(f"✓ uvx is installed")
        else:
            click.echo(f"✗ uvx is not available", err=True)
    except FileNotFoundError:
        click.echo(f"✗ uvx is not installed (required for MCP servers)", err=True)

    click.echo("\n✓ Dependency check complete\n")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
